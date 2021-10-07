from .. import BaseConfig, register_task

def filter_en(config, example):
    return example['language']=='en'


@register_task('lama')
class SST2Config(BaseConfig):
    task = 'lama'
    task_type = 'knowledge_probing'
    dataset_name = 'm_lama'
    metrics = 'accuracy'
    filter_fn = filter_en

