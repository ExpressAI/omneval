import argparse
import importlib
import os
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

TASK_REGISTRY = {}
PROCESSOR_REGISTRY = {}
EVALUATOR_REGISTRY = {}
METRICS_REGISTRY = {}


class BaseConfig(object):
    def __init__(self):
        assert getattr(self, 'task') is not None, "task name should be specified"
        assert getattr(self, 'task_type') is not None, "task type should be specified"
        assert getattr(self, 'metrics') is not None, "evaluation metrics should be specified"


class BaseProcessor(object):
    """The base processor class"""

    def __init__(self, arch, config):
        self.config = config
        self.raw_data = self.build_dataset()
        self.tokenizer = self.build_tokenizer(arch)

    def build_dataset(self):
        if hasattr(self.config, 'dataset_name'):
            dataset = self.config.dataset_name
        else:
            dataset = self.config.task
        return load_dataset(dataset)

    def build_tokenizer(self, arch):
        return AutoTokenizer.from_pretrained(arch)

    def generate_dataset(self, prompt_order=0):
        raise NotImplementedError

    def generate_aux_inputs(self, prompt_order=0):
        raise NotImplementedError

    @property
    def prompt_count(self):
        raise NotImplementedError

    @property
    def task_info(self):
        return self.config


class BaseEvaluator(object):

    def __init__(self, arch, config):
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.build_model(arch)
        self.tokenizer = self.build_tokenizer(arch)
        self.metrics_fn = build_metrics(config.metrics)

    def build_model(self, arch):
        raise NotImplementedError

    def build_tokenizer(self, arch):
        return AutoTokenizer.from_pretrained(arch)

    def eval(self, dataset, **kwargs):
        raise NotImplementedError


def build_config(task):
    return TASK_REGISTRY[task]()


def build_processor(arch, config):
    return PROCESSOR_REGISTRY[config.task_type](arch, config)


def build_evaluator(arch,  config):
    return EVALUATOR_REGISTRY[config.task_type][arch](arch, config)


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


from sklearn.metrics import accuracy_score

@register_metrics('accuracy')
def accuracy(labels, predictions):
    return accuracy_score(labels, predictions)




# automatically import any Python files in the models/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if not file.endswith('.py') and not file.startswith('_'):
        module = importlib.import_module('tasks.' + file)

