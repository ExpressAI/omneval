import argparse
import importlib
import os

from omneval.tasks import BaseConfig, BaseProcessor, BaseEvaluator

TASK_REGISTRY = {}
PROCESSOR_REGISTRY = {}
EVALUATOR_REGISTRY = {}
METRICS_REGISTRY = {}

def build_config(args, task):
    config = TASK_REGISTRY[task]()
    for k, v in vars(args).items():
        if k not in ['tasks', 'archs']:
            setattr(config, k, v)
    config.task = task
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


# automatically import any Python files in the models/ directory
for file in os.listdir(os.path.join(os.path.dirname(__file__), 'tasks')):
    if not file.endswith('.py') and not file.startswith('_'):
        module = importlib.import_module('omneval.tasks.' + file)