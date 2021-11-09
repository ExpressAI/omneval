from omneval.tasks import BaseConfig
from omneval.registry import register_task

@register_task('sem_eval_2010_task_8')
class SemEvalConfig(BaseConfig):
    # Required: The unique task identifier for this task
    task = 'sem_eval_2010_task_8'
    # Required: the task type, each task type corresponds to a data processor
    task_type = 'relation_extraction'
    # Required: Either input a file name that can be tracked in the environment(like  '${PATH_TO_FILE}/${FILE_NAME}')
    # or a str or list, which is a dataset name for huggingface's `datasets`
    dataset_name = ['sem_eval_2010_task_8']  # datasets.load_dataset('glue', 'sst2')
    # dataset_name = 'lama.json'  # datasets.load_dataset('json', 'lama.json')
    # dataset_name = 'lama '    # datasets.load_dataset('lama')
    # Required: The metrics used for this task
    metrics = 'f1'
    metrics_kwargs = {'average': 'micro'}
    # Optional: The data split used for evaluation: default 'test'
    test_subset = 'test'
    # Below are parameters for text classification
    # Required: prompt template:
    # e.g  `sentence` is the column name for the raw text "It was" and "." are templates, <mask> is the masked poition
    # Then the template is "<text> It was <mask>."
    templates = [
        'sentence|The|<mask>|<e1>|was|<mask>|to the|<mask>|<e2>|.',
    ]
    # Required: The label for this task
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    # Required: Manually-designed label for each class, the order of `labels` and label_mapping` should match
    label_mappings = [
        ['member', 'related', 'collection'],
        ['collection', 'related', 'member'],
        ['entity', 'related', 'origin'],
        ['origin', 'related', 'entity'],
        ['cause', 'related', 'effect'],
        ['effect', 'related', 'cause'],
        ['component', 'related', 'whole'],
        ['whole', 'related', 'component'],
        ['product', 'related', 'producer'],
        ['producer', 'related', 'product'],
        ['instrument', 'related', 'agency'],
        ['agency', 'related', 'instrument'],
        ['entity', 'related', 'destination'],
        ['destination', 'related', 'entity'],
        ['content', 'related', 'container'],
        ['container', 'related', 'content'],
        ['message', 'related', 'topic'],
        ['topic', 'related', 'message'],
        ['mention', 'irrelevant', 'mention'],
    ]
    # Optional: specify the name of the label column
    label_name = 'relation'
    sentence_label = 'sentence'
    remove_punc = False