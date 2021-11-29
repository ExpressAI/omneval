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
    metrics = 'rouge1'
    # Optional: The data split used for evaluation: default 'test'
    test_subset = 'test'
    label_name = 'highlights'

    # Below are parameters for text classification
    # Required: prompt template:
    # e.g  `sentence` is the column name for the raw text "It was" and "." are templates, <mask> is the masked poition
    # Then the template is "<text> It was <mask>."
    templates = [
        'article|In Summary,||highlights',
        'article|To Summarize,||highlights',
        'article|To conclude,||highlights',
        'Summarize the following article:|article||highlights',
    ]
    # Optional: choose the majority class of highest-topk label candidates
    eval_batch_size = 4
    decode_max_length = 128
    num_beams = 3
    max_seq_length = 512


@register_task('gigaword')
class SST2Config(BaseConfig):
    # Required: The unique task identifier for this task
    task = 'gigaword'
    # Required: the task type, each task type corresponds to a data processor
    task_type = 'generation'
    # Required: Either input a file name that can be tracked in the environment(like  '${PATH_TO_FILE}/${FILE_NAME}')
    # or a str or list, which is a dataset name for huggingface's `datasets`
    dataset_name = 'gigaword'  # datasets.load_dataset('glue', 'sst2')
    # dataset_name = 'lama.json'  # datasets.load_dataset('json', 'lama.json')
    # dataset_name = 'lama '    # datasets.load_dataset('lama')
    # Required: The metrics used for this task
    metrics = 'rouge1'
    # Optional: The data split used for evaluation: default 'test'
    test_subset = 'test'
    label_name = 'summary'

    # Below are parameters for text classification
    # Required: prompt template:
    # e.g  `sentence` is the column name for the raw text "It was" and "." are templates, <mask> is the masked poition
    # Then the template is "<text> It was <mask>."
    templates = [
        'document|In Summary,||summary',
        'document|To Summarize,||summary',
        'document|To conclude,||summary',
        'Summarize the following article:|document||summary',
    ]
    # Optional: choose the majority class of highest-topk label candidates
    eval_batch_size = 8
    decode_max_length = 48
    num_beams = 3
    max_seq_length = 128


@register_task('xsum')
class SST2Config(BaseConfig):
    # Required: The unique task identifier for this task
    task = 'xsum'
    # Required: the task type, each task type corresponds to a data processor
    task_type = 'generation'
    # Required: Either input a file name that can be tracked in the environment(like  '${PATH_TO_FILE}/${FILE_NAME}')
    # or a str or list, which is a dataset name for huggingface's `datasets`
    dataset_name = 'xsum'  # datasets.load_dataset('glue', 'sst2')
    # dataset_name = 'lama.json'  # datasets.load_dataset('json', 'lama.json')
    # dataset_name = 'lama '    # datasets.load_dataset('lama')
    # Required: The metrics used for this task
    metrics = 'rouge1'
    # Optional: The data split used for evaluation: default 'test'
    test_subset = 'test'
    label_name = 'summary'

    # Below are parameters for text classification
    # Required: prompt template:
    # e.g  `sentence` is the column name for the raw text "It was" and "." are templates, <mask> is the masked poition
    # Then the template is "<text> It was <mask>."
    templates = [
        'document|In Summary,||summary',
        'document|To Summarize,||summary',
        'document|To conclude,||summary',
        'Summarize the following article:|document||summary',
    ]
    # Optional: choose the majority class of highest-topk label candidates
    eval_batch_size = 4
    decode_max_length = 128
    num_beams = 3
    max_seq_length = 512


@register_task('samsum')
class SAMSUMConfig(BaseConfig):
    # Required: The unique task identifier for this task
    task = 'samsum'
    # Required: the task type, each task type corresponds to a data processor
    task_type = 'generation'
    # Required: Either input a file name that can be tracked in the environment(like  '${PATH_TO_FILE}/${FILE_NAME}')
    # or a str or list, which is a dataset name for huggingface's `datasets`
    dataset_name = 'samsum'  # datasets.load_dataset('glue', 'sst2')
    # dataset_name = 'lama.json'  # datasets.load_dataset('json', 'lama.json')
    # dataset_name = 'lama '    # datasets.load_dataset('lama')
    # Required: The metrics used for this task
    metrics = 'rouge1'
    # Optional: The data split used for evaluation: default 'test'
    test_subset = 'test'
    label_name = 'summary'

    # Below are parameters for text classification
    # Required: prompt template:
    # e.g  `sentence` is the column name for the raw text "It was" and "." are templates, <mask> is the masked poition
    # Then the template is "<text> It was <mask>."
    templates = [
        'dialogue|In Summary,||summary',
        'dialogue|To Summarize,||summary',
        'dialogue|To conclude,||summary',
        'Summarize the following dialogue:|dialogue||summary',
    ]
    # Optional: choose the majority class of highest-topk label candidates
    eval_batch_size = 4
    decode_max_length = 128
    num_beams = 3
    max_seq_length = 512