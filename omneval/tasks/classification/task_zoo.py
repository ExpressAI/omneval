from omneval.tasks import BaseConfig
from omneval.registry import register_task


@register_task('sst2')
class SST2Config(BaseConfig):
    # Required: The unique task identifier for this task
    task = 'sst2'
    # Required: the task type, each task type corresponds to a data processor
    task_type = 'classification'
    # Required: Either input a file name that can be tracked in the environment(like  '${PATH_TO_FILE}/${FILE_NAME}')
    # or a str or list, which is a dataset name for huggingface's `datasets`
    dataset_name = ['glue', 'sst2']  # datasets.load_dataset('glue', 'sst2')
    # dataset_name = 'lama.json'  # datasets.load_dataset('json', 'lama.json')
    # dataset_name = 'lama '    # datasets.load_dataset('lama')
    # Required: The metrics used for this task
    metrics = 'accuracy'
    # Optional: The data split used for evaluation: default 'test'
    test_subset = 'validation'

    # Below are parameters for text classification
    # Required: prompt template:
    # e.g  `sentence` is the column name for the raw text "It was" and "." are templates, <mask> is the masked poition
    # Then the template is "<text> It was <mask>."
    templates = [
        'sentence|It was |<mask>|.',
        'It was |<mask>|.|sentence|',
        'sentence|A |<mask>| movie.',
        'A |<mask>| movie.|sentence|',
        'sentence|<mask>|!',
        '<mask>|,|sentence|',
    ]
    # Required: The label for this task
    labels = [0, 1]
    # Required: Manually-designed label for each class, the order of `labels` and label_mapping` should match
    label_mappings = [
    ['bad', 'terrible', 'awful', 'dire', 'dreadful', 'fearful', 'horrible', 'abnormal', 'shocking', 'solemn',],
    ['great', 'good', 'right', 'sound', 'adorable', 'noble', 'pleasant', 'proper', 'decent', 'excellent']
    ]
    # Optional: choose the majority class of highest-topk label candidates
    topk = 7
    # Optional: choose the appropriate inference settings
    eval_batch_size = 32
    max_seq_length = 128

@register_task('mnli')
class MNLIConfig(BaseConfig):
    task = 'mnli'
    task_type = 'classification'
    dataset_name = ['glue', 'mnli']  # datasets.load_dataset('glue', 'sst2')
    metrics = 'accuracy'
    test_subset = 'validation_matched'
    templates = [
        'premise|?|<mask>|,|hypothesis|.|',
        'premise|,|<mask>|,|hypothesis|.|',
        'premise|!|<mask>|,|hypothesis|.|',
        '<mask>|,|premise|.|hypothesis|.|',
        '|premise|.|hypothesis|.|<mask>',
    ]
    labels = [2, 0, 1]
    label_mappings = [
    ['No', 'Meanwhile', 'But', 'Otherwise', 'Yet', 'Except', 'Alas', 'Conversely', 'However'],
    ['Yes', 'Exactly', 'Right', 'Absolutely', 'Yeah', 'Therefore', 'Definitely', 'Indeed', 'Yep'],
    ['Maybe', 'Probably', 'Perhaps', 'And', 'Possibly', 'Likely', 'Presumably', 'Also', 'Or'],
    ]
    # Optional: choose the majority class of highest-topk label candidates
    topk = 7
    remove_punc = True

    eval_batch_size = 32
    max_seq_length = 128

@register_task('snli')
class SNLIConfig(BaseConfig):
    task = 'mnli'
    task_type = 'classification'
    dataset_name = ['snli']  # datasets.load_dataset('glue', 'sst2')
    metrics = 'accuracy'
    test_subset = 'validation'
    templates = [
        'premise|?|<mask>|,|hypothesis|.|',
        'premise|,|<mask>|,|hypothesis|.|',
        'premise|!|<mask>|,|hypothesis|.|',
        '<mask>|,|premise|.|hypothesis|.|',
        '|premise|.|hypothesis|.|<mask>',
    ]
    labels = [2, 0, 1]
    label_mappings = [
    ['No', 'Meanwhile', 'But', 'Otherwise', 'Yet', 'Except', 'Alas', 'Conversely', 'However'],
    ['Yes', 'Exactly', 'Right', 'Absolutely', 'Yeah', 'Therefore', 'Definitely', 'Indeed', 'Yep'],
    ['Maybe', 'Probably', 'Perhaps', 'And', 'Possibly', 'Likely', 'Presumably', 'Also', 'Or'],
    ]
    # Optional: choose the majority class of highest-topk label candidates
    topk = 7
    remove_punc = True
    eval_batch_size = 32
    max_seq_length = 128

@register_task('mrpc')
class MRPC2Config(BaseConfig):
    task = 'mrpc'
    task_type = 'classification'
    dataset_name = ['glue', 'mrpc']
    test_subset = 'validation'
    metrics = 'f1'
    templates = [
        'premise|?|<mask>|,|hypothesis|.|',
        'premise|,|<mask>|,|hypothesis|.|',
        'premise|!|<mask>|,|hypothesis|.|',
        '<mask>|,|premise|.|hypothesis|.|',
        '|premise|.|hypothesis|.|<mask>',
    ]
    labels = [0, 1]
    label_mappings = [
    ['No', 'Meanwhile', 'But', 'Otherwise', 'Yet', 'Except', 'Alas', 'Conversely', 'However'],
    ['Yes', 'Exactly', 'Right', 'Absolutely', 'Yeah', 'Therefore', 'Definitely', 'Indeed', 'Yep'],
    ]
    # Optional: choose the majority class of highest-topk label candidates
    topk = 7
    eval_batch_size = 32
    max_seq_length = 128

@register_task('qqp')
class QQPConfig(BaseConfig):
    task = 'qqp'
    task_type = 'classification'
    dataset_name = ['glue', 'qqp']
    test_subset = 'validation'
    metrics = 'f1'
    templates = [
        'premise|?|<mask>|,|hypothesis|.|',
        'premise|,|<mask>|,|hypothesis|.|',
        'premise|!|<mask>|,|hypothesis|.|',
        '<mask>|,|premise|.|hypothesis|.|',
        '|premise|.|hypothesis|.|<mask>',
    ]
    labels = [0, 1]
    label_mappings = [
    ['No', 'Meanwhile', 'But', 'Otherwise', 'Yet', 'Except', 'Alas', 'Conversely', 'However'],
    ['Yes', 'Exactly', 'Right', 'Absolutely', 'Yeah', 'Therefore', 'Definitely', 'Indeed', 'Yep'],
    ]
    topk = 7
    eval_batch_size = 32
    max_seq_length = 128


@register_task('qnli')
class QQPConfig(BaseConfig):
    task = 'qnli'
    task_type = 'classification'
    dataset_name = ['glue', 'qnli']
    test_subset = 'validation'
    metrics = 'accuracy'
    templates = [
        'premise|<mask>|,|hypothesis|',
        '<mask>|,|premise|hypothesis|',
        'premise|hypothesis|<mask>',
    ]
    labels = [0, 1]
    label_mappings = [
    ['No', 'Meanwhile', 'But', 'Otherwise', 'Yet', 'Except', 'Alas', 'Conversely', 'However'],
    ['Yes', 'Exactly', 'Right', 'Absolutely', 'Yeah', 'Therefore', 'Definitely', 'Indeed', 'Yep'],
    ]
    topk = 7
    remove_punc = True
    eval_batch_size = 32
    max_seq_length = 128


@register_task('rte')
class RTEConfig(BaseConfig):
    task = 'rte'
    task_type = 'classification'
    dataset_name = ['glue', 'rte']
    test_subset = 'validation'
    metrics = 'f1'
    templates = [
        'sentence1|?|<mask>|,|sentence2|.|',
        'sentence1|,|<mask>|,|sentence2|.|',
        'sentence1|!|<mask>|,|sentence2|.|',
        '<mask>|,|premise|.|sentence2|.|',
        '|sentence1|.|sentence2|.|<mask>',
    ]
    labels = [0, 1]
    label_mappings = [
    ['No', 'Meanwhile', 'But', 'Otherwise', 'Yet', 'Except', 'Alas', 'Conversely', 'However'],
    ['Yes', 'Exactly', 'Right', 'Absolutely', 'Yeah', 'Therefore', 'Definitely', 'Indeed', 'Yep'],
    ]
    # Optional: choose the majority class of highest-topk label candidates
    topk = 7
    remove_punc = True
    eval_batch_size = 32
    max_seq_length = 128
