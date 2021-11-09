from omneval.tasks import BaseConfig
from omneval.registry import register_task
@register_task('cnn_dailymail')
class SST2Config(BaseConfig):
    # Required: The unique task identifier for this task
    task = 'cnn_dailymail'
    # Required: the task type, each task type corresponds to a data processor
    task_type = 'generation'
    # Required: Either input a file name that can be tracked in the environment(like  '${PATH_TO_FILE}/${FILE_NAME}')
    # or a str or list, which is a dataset name for huggingface's `datasets`
    dataset_name = ['cnn_dailymail', '3.0.0']  # datasets.load_dataset('glue', 'sst2')
    # dataset_name = 'lama.json'  # datasets.load_dataset('json', 'lama.json')
    # dataset_name = 'lama '    # datasets.load_dataset('lama')
    # Required: The metrics used for this task
    metrics = 'rouge'
    # Optional: The data split used for evaluation: default 'test'
    test_subset = 'test'
    label_name = 'highlights'

    # Below are parameters for text classification
    # Required: prompt template:
    # e.g  `sentence` is the column name for the raw text "It was" and "." are templates, <mask> is the masked poition
    # Then the template is "<text> It was <mask>."
    templates = [
        'article|In Summary,||highlights',
    ]
    # Optional: choose the majority class of highest-topk label candidates
    topk = 1
    eval_batch_size = 32
