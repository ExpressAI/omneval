from omneval.tasks import BaseConfig
from omneval.registry import register_task

@register_task('lama')
class LAMAConfig(BaseConfig):
    task = 'lama'
    task_type = 'knowledge_probing'
    dataset_name = 'datasets/lama/lama.json'
    metrics = 'accuracy'
    label_name = 'obj_label'
    eval_batch_size = 32
    templates = ['Specified for each sample']
    max_seq_length = 128
    # For the lama task, you should define the unified vocab for all PLMs
    vocab_file = 'datasets/lama/cased.vocab'


