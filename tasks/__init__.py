import argparse
import importlib
import os
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import pdb
import collections

TASK_REGISTRY = {}
PROCESSOR_REGISTRY = {}
EVALUATOR_REGISTRY = {}
METRICS_REGISTRY = {}


def collate_fn(batch, exclude=[]):
    keys = batch[0].keys()
    return {k: (torch.LongTensor([bz[k] for bz in batch]) if k not in exclude else [bz[k] for bz in batch]) for k in keys}


def get_logits(outputs):
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    elif hasattr(outputs, 'prediction_logits'):
        logits = outputs.prediction_logits
    else:
        raise NotImplementedError
    return logits


class BaseConfig(object):
    def __init__(self):
        assert getattr(self, 'task') is not None, "task name should be specified"
        assert getattr(self, 'task_type') is not None, "task type should be specified"
        assert getattr(self, 'metrics') is not None, "evaluation metrics should be specified"
        self.eval_batch_size = getattr(self, 'eval_batch_size', 8)


class BaseProcessor(object):
    """The base processor class"""

    def __init__(self, config):
        self.config = config    # task config
        self.raw_data = self.build_dataset()
        self.tokenizer = self.build_tokenizer()
        self.label_name = getattr(self.config, 'label_name', 'label')
        self.padding_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        self.mask_token = self.tokenizer.mask_token if self.tokenizer._mask_token is not None else self.tokenizer.unk_token
        self.mask_token_id = self.tokenizer.mask_token_id if self.tokenizer._mask_token is not None else self.tokenizer.unk_token_id
        self.max_seq_length = getattr(self.config, "max_seq_length", 512)

    def build_dataset(self):
        """Import the raw dataset"""
        if hasattr(self.config, 'dataset_name'):
            dataset = self.config.dataset_name
        else:
            dataset = self.config.task
        test_subset = getattr(self.config, 'test_subset', 'test')
        if isinstance(dataset, str):
            if os.path.exists(dataset):
                logging.info("Load dataset from local files: %s"%dataset)
                fmt = dataset.split('.')[-1]
                df = load_dataset(fmt, data_files={test_subset: dataset}, split=test_subset)
            else:
                logging.info("Search dataset %s from Huggingface's Hub"%dataset)
                df = load_dataset(dataset, split=test_subset)
        else:
            logging.info("Search dataset %s from Huggingface's Hub" % '/'.join(dataset))
            df = load_dataset(*dataset, split=test_subset)
        if hasattr(self.config, 'filter_fn'):
            filter_fn = getattr(self.config, 'filter_fn', None)
            assert callable(filter_fn)
        try:
            df = df.filter(filter_fn)
            logging.info("Use filter_fn to filter the dataset, got %d examples"%(df.num_rows))
        except:
            logging.info("No filter_fn or filter_fn not valid, use the original data, got %d examples"%(df.num_rows))
        return df

    def build_tokenizer(self):
        """Build the tokenizer given model arch name"""
        return AutoTokenizer.from_pretrained(self.config.arch)

    def generate_dataset(self, pid=0):
        """Prompting each instance and build dataset directly for the Evaluator"""
        prompt_schema = self.config.templates[pid]
        remove_columns = difference(self.raw_data.features.keys(), self.label_name)

        return self.raw_data.map(
            lambda x: self.prompting(example=x,
                                     prompt_schema=prompt_schema),
                                     remove_columns=remove_columns)


    def prompting(self, example, prompt_schema):
        raise NotImplementedError

    def generate_aux_inputs(self, pid=0):
        """Generate other inputs required for the dataset"""
        return {}

    @property
    def prompt_count(self):
        """Count for number of prompt schema"""
        raise NotImplementedError

    @property
    def task_info(self):
        """Print the task info"""
        return self.config

    def prompt_schema(self, pid):
        raise NotImplementedError


class BaseEvaluator(object):

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.arch = config.arch
        self.model = self.build_model()
        self.tokenizer = self.build_tokenizer()
        self.metrics_fn = build_metrics(config.metrics)
        self.padding_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        self.mask_token = self.tokenizer.mask_token if self.tokenizer._mask_token is not None else self.tokenizer.unk_token
        self.mask_token_id = self.tokenizer.mask_token_id if self.tokenizer._mask_token is not None else self.tokenizer.unk_token_id
        self.label_name = getattr(self.config, 'label_name', 'label')


    def build_model(self):
        """Initialize model for evaluation"""
        raise NotImplementedError

    def build_tokenizer(self):
        """
        Build Tokenizer for the evaluator
        :param arch: model arch
        :return: the tokenizer
        """
        return AutoTokenizer.from_pretrained(self.config.arch)


    def preprocessing(self, dataset, **kwargs):
        """Preprocessing the dataset and other auxiliary inputs"""
        return dataset, kwargs


    def decode(self, batch, **kwargs):
        """
        Generate the predictions for each test instance
        :param dataset: dataset generated by the evaluator's `preprocessing`
        :param kwargs: other input generated by the evaluator's `preprocessing`
        :return: a dictionary, which a required key `prediction` for
        """
        raise NotImplementedError


    def eval(self, dataset, **kwargs):
        """
        Generate the evaluation metrics and analysis (call: self.decode function)
        :param dataset: dataset generated by the processor's `generate_dataset`
        :param kwargs: other input generated by the processor's `generate_aux_input`
        :return: The evaluation metrics
        """
        self.model.eval()
        dataset, kwargs = self.preprocessing(dataset, **kwargs)
        label_name = getattr(self.config, 'label_name', 'label')
        dataloader = DataLoader(dataset, batch_size=self.config.eval_batch_size, collate_fn=collate_fn)
        labels = []
        res = collections.defaultdict(list)
        for batch in tqdm(dataloader):
            label = batch.pop(label_name).view(-1).cpu().detach().tolist()
            batch = {k: v.to(self.device) for k, v in batch.items()}
            predictions = self.decode(batch, **kwargs)
            labels += label
            merge_fn(res, predictions)
        metrics = self.metrics_fn(labels, res['predictions'])
        res['label'] = labels
        logging.info("Evaluation metrics of the task %s on model: %s---%s: %.3f" % (
            self.config.task, self.arch, self.config.metrics, metrics))
        return res

    def write_to_json(self, outputs, output_dir):
        res = []
        for output in outputs:
            length = len(output['predictions'])
            for i in range(length):
                res.append(self.parse_predictions({k: v[i] for k, v in output.items()}))
        with open(output_dir, 'w') as f:
            for row in res:
                f.writelines(str(row)+'\n')


    def parse_predictions(self, prediction):
        raise NotImplementedError


def build_config(args):
    config = TASK_REGISTRY[args.task]()
    for k, v in vars(args).items():
        setattr(config, k, v)
    return config


def build_processor(config):
    return PROCESSOR_REGISTRY[config.task_type](config)


def build_evaluator(config):
    return EVALUATOR_REGISTRY[config.task_type][config.arch](config)


def build_metrics(name):
    return METRICS_REGISTRY[name]


def register_task(name):
    """
    Configurations of the new task can be added with the :func:`register_task`
    function decorator.

    For example::

        @register_model('sst2')
        class SST2Config(BaseConfig):
            (...)
    Args:
        name (str): the name of the model
    """

    def register_task_cls(cls):
        if name in TASK_REGISTRY:
            raise ValueError('Cannot register duplicate task ({})'.format(name))
        if not issubclass(cls, BaseConfig):
            raise ValueError('Task ({}: {}) must extend from the BaseConfig'.format(name, cls.__name__))
        TASK_REGISTRY[name] = cls
        return cls

    return register_task_cls


def register_processor(task_type):
    """
    New data processor can be added with the :func:`register_processor`
    """

    def register_processor_fn(cls):
        if task_type in PROCESSOR_REGISTRY:
            raise ValueError('Cannot register duplicate processor ({})'.format(task_type))
        if not issubclass(cls, BaseProcessor):
            raise ValueError('Task ({}: {}) must extend from the BaseProcessor'.format(task_type, cls.__name__))
        PROCESSOR_REGISTRY[task_type] = cls
        return cls

    return register_processor_fn


def register_evaluator(task_type, archs):
    """
    New evaluator can be added with the :func:`register_processor`
    """
    def register_evaluator_fn(cls):
        if not issubclass(cls, BaseEvaluator):
            raise ValueError('Task ({}: {}) must extend from the BaseEvaluator'.format(task_type, cls.__name__))
        if EVALUATOR_REGISTRY.get(task_type) is None:
            EVALUATOR_REGISTRY[task_type] = {}

        for arch in archs:
            if arch in EVALUATOR_REGISTRY[task_type]:
                raise ValueError('Cannot register duplicate evaluator ({})'.format(arch))
            EVALUATOR_REGISTRY[task_type][arch] = cls
        return cls

    return register_evaluator_fn


def register_metrics(name):
    """
    New metrics can be added with the :func:`register_metrics`
    """
    def register_metrics_fn(fn):
        if name in METRICS_REGISTRY:
            raise ValueError('Cannot register duplicate metrics ({})'.format(name))
        if not callable(fn):
            raise ValueError('metrics must be callable ({})'.format(name))
        METRICS_REGISTRY[name] = fn
        return fn

    return register_metrics_fn


from sklearn.metrics import accuracy_score, f1_score

@register_metrics('accuracy')
def accuracy(labels, predictions):
    return accuracy_score(labels, predictions)

@register_metrics('f1')
def f1(labels, predictions):
    return f1_score(labels, predictions)

def merge_fn(dict1, dict2):
    for k, v in dict2.items():
        dict1[k] += v

def difference(list1, list2):
    return [item for item in list1 if item not in list2]


# automatically import any Python files in the models/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if not file.endswith('.py') and not file.startswith('_'):
        module = importlib.import_module('tasks.' + file)



