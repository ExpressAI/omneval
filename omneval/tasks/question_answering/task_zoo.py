from omneval.tasks import BaseConfig
from omneval.registry import register_task


@register_task('squad')
class SquadConfig(BaseConfig):
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
    eval_batch_size = 8
    decode_max_length = 10
    num_beams = 3
    max_seq_length = 512


@register_task('squad_v2')
class SquadConfig(BaseConfig):
    # Required: The unique task identifier for this task
    task = 'squad_v2'
    # Required: the task type, each task type corresponds to a data processor
    task_type = 'question_answering'
    # Required: Either input a file name that can be tracked in the environment(like  '${PATH_TO_FILE}/${FILE_NAME}')
    # or a str or list, which is a dataset name for huggingface's `datasets`
    dataset_name = 'squad_v2' # datasets.load_dataset('glue', 'sst2')
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
    eval_batch_size = 8
    decode_max_length = 10
    num_beams = 3
    max_seq_length = 512


@register_task('adversarial_qa')
class SquadConfig(BaseConfig):
    # Required: The unique task identifier for this task
    task = 'adversarial_qa'
    # Required: the task type, each task type corresponds to a data processor
    task_type = 'question_answering'
    # Required: Either input a file name that can be tracked in the environment(like  '${PATH_TO_FILE}/${FILE_NAME}')
    # or a str or list, which is a dataset name for huggingface's `datasets`
    dataset_name = ['adversarial_qa', 'adversarialQA'] # datasets.load_dataset('glue', 'sst2')
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
    eval_batch_size = 8
    decode_max_length = 10
    num_beams = 3
    max_seq_length = 512


@register_task('ropes')
class SquadConfig(BaseConfig):
    # Required: The unique task identifier for this task
    task = 'ropes'
    # Required: the task type, each task type corresponds to a data processor
    task_type = 'question_answering'
    # Required: Either input a file name that can be tracked in the environment(like  '${PATH_TO_FILE}/${FILE_NAME}')
    # or a str or list, which is a dataset name for huggingface's `datasets`
    dataset_name = 'ropes' # datasets.load_dataset('glue', 'sst2')
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
        'background|situation|question||answers',
        'situation|background|question||answers',
    ]
    # Optional: choose the majority class of highest-topk label candidates
    topk = 1
    eval_batch_size = 8
    decode_max_length = 10
    num_beams = 3
    max_seq_length = 512


@register_task('subjqa')
class SquadConfig(BaseConfig):
    # Required: The unique task identifier for this task
    task = 'subjqa'
    # Required: the task type, each task type corresponds to a data processor
    task_type = 'question_answering'
    # Required: Either input a file name that can be tracked in the environment(like  '${PATH_TO_FILE}/${FILE_NAME}')
    # or a str or list, which is a dataset name for huggingface's `datasets`
    dataset_name = 'subjqa' # datasets.load_dataset('glue', 'sst2')
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
    eval_batch_size = 8
    decode_max_length = 64
    num_beams = 3
    max_seq_length = 512

def processing_narrativeqa(config, data):
    def merge_options(example):
        example['summary'] = example['document']['summary']['text']
        example['question'] = example['question']['text']
        example['id'] = example['summary'][:10]
        example['answers'] = example['answers'][0]
        return example
    return data.map(merge_options, remove_columns=['document'])


@register_task('narrativeqa')
class SquadConfig(BaseConfig):
    # Required: The unique task identifier for this task
    task = 'narrativeqa'
    # Required: the task type, each task type corresponds to a data processor
    task_type = 'question_answering'
    # Required: Either input a file name that can be tracked in the environment(like  '${PATH_TO_FILE}/${FILE_NAME}')
    # or a str or list, which is a dataset name for huggingface's `datasets`
    dataset_name = ['narrativeqa'] # datasets.load_dataset('glue', 'sst2')
    # dataset_name = 'lama.json'  # datasets.load_dataset('json', 'lama.json')
    # dataset_name = 'lama '    # datasets.load_dataset('lama')
    # Required: The metrics used for this task
    metrics = 'squad'
    # Optional: The data split used for evaluation: default 'test'
    test_subset = 'test'
    label_name = 'answers'
    remain_columns = ['id', 'answers']
    # Below are parameters for text classification
    # Required: prompt template:
    # e.g  `sentence` is the column name for the raw text "It was" and "." are templates, <mask> is the masked poition
    # Then the template is "<text> It was <mask>."
    templates = [
        'summary|question||answers',
    ]
    # Optional: choose the majority class of highest-topk label candidates
    topk = 1
    eval_batch_size = 8
    decode_max_length = 10
    num_beams = 3
    max_seq_length = 512
    data_preprocessing = processing_narrativeqa