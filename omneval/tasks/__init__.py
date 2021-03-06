import argparse
import importlib
import os
import torch
from transformers import AutoTokenizer, T5Tokenizer, T5TokenizerFast
from datasets import load_dataset
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import collections
from omneval.utils import collate_fn, get_logits, pad_input_ids, difference, merge_fn, adjust_length, get_masked_tokens


class BaseConfig(object):
    """The base configuration class"""

    def __init__(self):
        assert getattr(self, 'task') is not None, "task name should be specified"
        assert getattr(self, 'task_type') is not None, "task type should be specified"
        assert getattr(self, 'metrics') is not None, "evaluation metrics should be specified"
        self.eval_batch_size = getattr(self, 'eval_batch_size', 8)
        self.metrics_kwargs = getattr(self, 'metrics_kwargs', {})
        self.max_seq_length = getattr(self, 'max_seq_length', 512)
        self.label_name = getattr(self, 'label_name', 'label')


class BaseProcessor(object):
    """The base processor class"""

    def __init__(self, config):
        self.config = config  # task config
        self.raw_data = self.build_dataset()
        self.tokenizer = self.build_tokenizer()
        self.label_name = self.config.label_name
        self.padding_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        self.mask_token, self.mask_token_id = get_masked_tokens(config, self.tokenizer)
        self.max_seq_length = adjust_length(self.config)

    def build_dataset(self):
        """Import the raw dataset"""
        if hasattr(self.config, 'dataset_name'):
            dataset = self.config.dataset_name
        else:
            dataset = self.config.task
        test_subset = getattr(self.config, 'test_subset', 'test')
        if isinstance(dataset, str):
            # load dataset from local files, support format: json, csv, plain text,
            # use huggingface.datasets.load_dataset API
            if os.path.exists(dataset):
                logging.info("Load dataset from local files: %s" % dataset)
                fmt = dataset.split('.')[-1]
                if fmt not in ('json', 'csv'):
                    logging.info("`load_dataset` does not support this data format, "
                                 "use the plain `text` format to import")
                    fmt = 'text'
                df = load_dataset(fmt, data_files={test_subset: dataset}, split=test_subset)
            else:
                logging.info("Search dataset %s from Huggingface's Hub" % dataset)
                df = load_dataset(dataset, split=test_subset)
        else:
            # some dataset names have multiple arguments
            logging.info("Search dataset %s from Huggingface's Hub" % '/'.join(dataset))
            df = load_dataset(*dataset, split=test_subset)
        # Options for user-defined filter functions
        if hasattr(self.config, 'filter_fn'):
            filter_fn = getattr(self.config, 'filter_fn', None)
        try:
            df = df.filter(filter_fn)
            logging.info("Use config.filter_fn to filter the dataset, got %d examples" % (df.num_rows))
        except:
            logging.info("No config.filter_fn or config.filter_fn not valid, use the original data, got %d examples" % (df.num_rows))

        # Options for user-defined dataset modifications
        if hasattr(self.config, 'data_preprocessing'):

            data_preprocessing = getattr(self.config, 'data_preprocessing', None)
        try:
            df = data_preprocessing(df)
            logging.info("Use config.data_preprocessing to preprocess the dataset, got %d examples" % (df.num_rows))
        except:
            logging.info("No config.data_preprocessing or config.data_preprocessing not valid, "
                         "use the original data, got %d examples" % (df.num_rows))
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

        # print one example for testing:
        assert len(self.raw_data) > 0, "the raw dataset cannot be empty!"
        test_example = self.raw_data[0]
        logging.info("An example prompts of pid=%s: ")


        return self.raw_data.map(
            lambda x: self.prompting(example=x,
                                     prompt_schema=prompt_schema,
                                     max_length=self.max_seq_length - prompt_length),
            remove_columns=remove_columns)

    def prompting(self, example, prompt_schema, max_length=512):
        raise NotImplementedError

    def generate_aux_inputs(self, pid=0):
        """Generate other inputs required for the dataset"""
        return {}

    # TODO: Not compatible with the RE evaluater, need to check this
    def generate_calibrate_example(self, pid):
        """Generate calibrated examples"""
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
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.padding_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        self.mask_token, self.mask_token_id = get_masked_tokens(config, self.tokenizer)
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
        :return: a dictionary, which a required key `prediction` for
        """
        raise NotImplementedError

    def eval(self, dataset, metrics_fn, **kwargs):
        """
        Generate the evaluation metrics and analysis (call: self.decode function)
        :param dataset: dataset generated by the processor's `generate_dataset`
        :param metrics_fn: metrics used for evaluation
        :param kwargs: other input generated by the processor's `generate_aux_input`
        :return: The evaluation metrics
        """
        self.model.eval()
        dataset, kwargs = self.preprocessing(dataset, **kwargs)
        dataloader = DataLoader(dataset, batch_size=self.config.eval_batch_size,
                                collate_fn=lambda x: collate_fn(x, exclude=self.exclude_collate_features))
        labels = []
        res = collections.defaultdict(list)
        for batch in tqdm(dataloader):
            label = batch.pop(self.label_name)
            batch = {k: v.to(self.device) if k not in self.exclude_collate_features else v for k, v in batch.items()}
            predictions = self.decode(batch, **kwargs)
            labels += label
            merge_fn(res, predictions)
        predictions = self.process_predictions_for_metrics(res)
        metrics = metrics_fn.compute(predictions=predictions, references=labels,
                                    **self.config.metrics_kwargs)
        res['label'] = labels
        res_list = []
        length = len(res['predictions'])
        for i in range(length):
            res_list.append(self.parse_predictions({k: v[i] for k, v in res.items()}))
        eval_result = {'plm': self.config.arch}
        eval_result.update(metrics)
        eval_result.update(self.analysis(res_list))
        return res_list, eval_result

    def process_predictions_for_metrics(self, res):
        """process prediction for metrics"""
        return res['predictions']

    def write_inference_to_json(self, res, pid):
        """write processed prediction into json file"""
        template = self.config.templates[pid]
        arch = self.config.arch.split('/')[-1]
        output_filename = self.config.task + '_' + arch + '_' + template + '.json'
        output_filename = os.path.join(self.config.out_dir, output_filename)
        with open(output_filename, 'w') as f:
            for row in res:
                f.writelines(str(row) + '\n')

    def parse_predictions(self, prediction):
        """post-process predictions"""
        raise NotImplementedError

    def analysis(self, res_list):
        """optional: extra analysis for predictions beyond metrics"""
        return {}

    @property
    def exclude_collate_features(self):
        return [self.label_name]

