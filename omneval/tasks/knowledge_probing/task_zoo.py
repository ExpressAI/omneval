from omneval.tasks import BaseConfig
from omneval.registry import register_task


@register_task('lama')
class LAMAConfig(BaseConfig):
    task = 'lama'
    task_type = 'knowledge_probing'
    dataset_name = 'lama.json'
    metrics = 'accuracy'
    label_name = 'obj_label'
    eval_batch_size = 32
    templates = ['Specified for each sample']

