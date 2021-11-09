import argparse
import importlib
import os
import torch
from transformers import AutoTokenizer
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import collections
from omneval.utils import collate_fn, get_logits, pad_input_ids, difference, merge_fn
import pdb

class BaseConfig(object):
    def __init__(self):
        assert getattr(self, 'task') is not None, "task name should be specified"
        assert getattr(self, 'task_type') is not None, "task type should be specified"
        assert getattr(self, 'metrics') is not None, "evaluation metrics should be specified"
        self.eval_batch_size = getattr(self, 'eval_batch_size', 8)
        self.metrics_kwargs = getattr(self, 'metrics_kwargs', {})


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
        self.max_seq_length = getattr(self.config, "max_seq_length", 64)

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
        calibrate_word = self.generate_calibrate_example(pid)
        prompt_length = sum(calibrate_word['attention_mask'])
        return self.raw_data.map(
            lambda x: self.prompting(example=x,
                                     prompt_schema=prompt_schema,
                                     max_length=self.max_seq_length-prompt_length),
                                     remove_columns=remove_columns)

    def prompting(self, example, prompt_schema, max_length=512):
        raise NotImplementedError

    def generate_aux_inputs(self, pid=0):
        """Generate other inputs required for the dataset"""
        return {}

    def generate_calibrate_example(self, pid):
        prompt_schema = self.prompt_schema(pid)
        null_example = {k: '' for k in self.raw_data.features.keys()}
        return self.prompting(null_example, prompt_schema)

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
        self.metrics_fn = load_metric(config.metrics)
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
        dataloader = DataLoader(dataset, batch_size=self.config.eval_batch_size, collate_fn=lambda x: collate_fn(x, exclude=[self.label_name]))
        labels = []
        res = collections.defaultdict(list)
        for batch in tqdm(dataloader):
            label = batch.pop(self.label_name)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            predictions = self.decode(batch, **kwargs)
            labels += label
            merge_fn(res, predictions)
        metrics = self.metrics_fn.compute(predictions=res['predictions'], references=labels, **self.config.metrics_kwargs)
        res['label'] = labels
        res_list = []
        length = len(res['predictions'])
        for i in range(length):
            res_list.append(self.parse_predictions({k: v[i] for k, v in res.items()}))
        eval_result = {'plm': self.config.arch}
        eval_result.update(metrics)
        eval_result.update(self.analysis(res_list))
        return res_list, eval_result

    def write_inference_to_json(self, res, pid):
        arch = self.config.arch.split('/')[-1]
        output_filename = self.config.task+'_'+arch+'_'+str(pid)+'.json'
        output_filename = os.path.join(self.config.out_dir, output_filename)
        with open(output_filename, 'w') as f:
            for row in res:
                f.writelines(str(row)+'\n')

    def parse_predictions(self, prediction):
        raise NotImplementedError

    def analysis(self, res_list):
        raise NotImplementedError





