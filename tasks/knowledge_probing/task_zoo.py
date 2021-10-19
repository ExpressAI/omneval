from .. import BaseConfig, register_task

# def filter_en(config, example):
#     return example['language']=='en'


@register_task('lama')
class LAMAConfig(BaseConfig):
    task = 'lama'
    task_type = 'knowledge_probing'
    dataset_name = 'lama.json'
    metrics = 'accuracy'
    label_name = 'obj_label'
    eval_batch_size = 32

