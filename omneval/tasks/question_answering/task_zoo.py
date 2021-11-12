from omneval.tasks import BaseConfig
from omneval.registry import register_task


@register_task('squad')
class SST2Config(BaseConfig):
    # Required: The unique task identifier for this task
    task = 'squad'
    # Required: the task type, each task type corresponds to a data processor
    task_type = 'question_answering'
    # Required: Either input a file name that can be tracked in the environment(like  '${PATH_TO_FILE}/${FILE_NAME}')
    # or a str or list, which is a dataset name for huggingface's `datasets`
    dataset_name = 'squad' # datasets.load_dataset('glue', 'sst2')
    # dataset_name = 'lama.json'  # datasets.load_dataset('json', 'lama.json')
    # dataset_name = 'lama '    # datasets.load_dataset('lama')
    # Required: The metrics used for this task
    metrics = 'squad'
    # Optional: The data split used for evaluation: default 'test'
    test_subset = 'validation'
    label_name = 'answers'
    remain_columns = ['id', 'answers']
    # Below are parameters for text classification
    # Required: prompt template:
    # e.g  `sentence` is the column name for the raw text "It was" and "." are templates, <mask> is the masked poition
    # Then the template is "<text> It was <mask>."
    templates = [
        'context|question||answers',
    ]
    # Optional: choose the majority class of highest-topk label candidates
    topk = 1
    eval_batch_size = 16
    decode_max_length = 10
    num_beams = 3
    max_seq_length = 512
