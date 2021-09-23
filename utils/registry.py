import argparse
import importlib
import os

from ..tasks import BaseConfig, BaseProcessor, BaseEvaluator

TASK_REGISTRY = {}
PROCESSOR_REGISTRY = {}
EVALUATOR_REGISTRY = {}


def build_config(args):
    return TASK_REGISTRY[args.task]()

def build_processor(config):
    return PROCESSOR_REGISTRY[config.task](config)

def build_evaluator(arch,  config):
    return EVALUATOR_REGISTRY[config.task_type][arch](arch, config)


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
            raise ValueError('Cannot register duplicate processor ({})'.format(model_name))
        if not issubclass(cls, BaseProcessor):
            raise ValueError('Task ({}: {}) must extend from the BaseProcessor'.format(task_type, cls.__name__))
        PROCESSOR_REGISTRY[task_type] = cls
        return cls

    return register_processor_fn


def register_evaluator(task_type, arch_name):
    """
    New evaluator can be added with the :func:`register_processor`
    """
    def register_evaluator_fn(cls):
        if task_type not in PROCESSOR_REGISTRY:
            raise ValueError('Cannot register evaluators for unknown task types ({})'.format(task_type))
        if not issubclass(cls, BaseEvaluator):
            raise ValueError('Task ({}: {}) must extend from the BaseEvaluator'.format(task_type, cls.__name__))
        if EVALUATOR_REGISTRY.get(task_type) is None:
            EVALUATOR_REGISTRY[task_type] = {}
        if isinstance(arch_name, str):
            arch_name = [arch_name, ]
        for arch in arch_name:
            if arch in EVALUATOR_REGISTRY[task_type]:
                raise ValueError('Cannot register duplicate evaluator ({})'.format(arch))
            EVALUATOR_REGISTRY[task_type][arch] = cls
        return cls

    return register_evaluator_fn


# automatically import any Python files in the models/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        model_name = file[:file.find('.py')]
        module = importlib.import_module('fairseq.models.' + model_name)

        # extra `model_parser` for sphinx
        if model_name in TASK_REGISTRY:
            parser = argparse.ArgumentParser(add_help=False)
            group_archs = parser.add_argument_group('Named architectures')
            group_archs.add_argument('--arch', choices=ARCH_MODEL_INV_REGISTRY[model_name])
            group_args = parser.add_argument_group('Additional command-line arguments')
            TASK_REGISTRY[model_name].add_args(group_args)
            globals()[model_name + '_parser'] = parser
