from .. import BaseConfig, register_task


@register_task('sst2')
class SST2Config(BaseConfig):
    task = 'sst2'
    task_type = 'classification'
    dataset_name = ['glue', 'sst2']
    test_subset = 'validation'
    metrics = 'accuracy'
    templates = [
        'sentence|It was |<mask>|.'
    ]
    labels = [0, 1]
    remove_columns = ['sentence']
    label_mappings = [('terrible', 'great')]


@register_task('mrpc')
class MRPC2Config(BaseConfig):
    task = 'mrpc'
    task_type = 'classification'
    dataset_name = ['glue', 'mrpc']
    test_subset = 'validation'
    metrics = 'f1'
    templates = [
        'sentence1|?|<mask>|,|sentence2|.|',
    ]
    labels = [0, 1]
    remove_columns = ['sentence1', 'sentence2','idx']
    label_mappings = [('No', 'Yes')]


@register_task('qqp')
class QQPConfig(BaseConfig):
    task = 'qqp'
    task_type = 'classification'
    dataset_name = ['glue', 'qqp']
    test_subset = 'validation'
    metrics = 'f1'
    templates = [
        'sentence1|?|<mask>|,|sentence2|.|',
    ]
    labels = [0, 1]
    remove_columns = ['sentence1', 'sentence2','idx']
    label_mappings = [('No', 'Yes')]


# 'sst2' is the unique task identifier for this task
@register_task('sst2_m')
class SST2Config(BaseConfig):
    # Required: The unique task identifier for this task
    task = 'sst2_m'
    # Required: the task type, each task type corresponds to a data processor
    task_type = 'classification_m'
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
        'sentence|It was |<mask>|.'
    ]
    # Required: The label for this task
    labels = [0, 1]
    # Required: Manually-designed label for each class, the order of `labels` and label_mapping` should match
    q = [
    ['bad', 'terrible', 'awful', 'dire', 'dread', 'dreadful', 'fearful', 'grand', 'horrible', 'imposing', 'majestic', 'shocking', 'solemn',],
    ['great', 'good', 'right', 'sound', 'pious', 'benevolent', 'competent', 'real', 'considerable', 'righteous', 'proper', 'upright', 'excellent']
    ]
    # Optional: choose the majority class of highest-topk label candidates
    topk = 7


@register_task('snli')
class SNLIConfig(BaseConfig):
    task = 'snli'
    task_type = 'classification'
    dataset_name = 'snli'
    test_subset = 'validation'
    metrics = 'accuracy'
    templates = [
        'premise|?|<mask>|,|hypothesis|.|',
    ]
    labels = [2, 0, 1]
    remove_columns = ['premise', 'hypothesis']
    label_mappings = [('No', 'Yes', 'Maybe'),]
    remove_punc = True
    eval_batch_size = 32


@register_task('snli_m')
class SNLIConfig(BaseConfig):
    task = 'snli_m'
    task_type = 'classification_m'
    dataset_name = 'snli'
    test_subset = 'validation'
    metrics = 'accuracy'
    templates = [
        'premise|?|<mask>|,|hypothesis|.|',
    ]
    labels = [2, 0, 1]
    remove_columns = ['premise', 'hypothesis']
    label_mappings = [['No', 'Instead', 'Normally', 'Meanwhile', 'Unless', 'Otherwise', 'Except', 'Plus'],
                      ['Yes','Indeed', 'YES', 'Right', 'Regardless', 'Ok', 'Alright', 'Sometimes'],
                      ['Maybe', 'Seriously',
                       'This',
                       'Specifically',
                       'Fortunately',
                       'Specifically',
                       'Watch',
                       'Hopefully']
]
    remove_punc = True
    eval_batch_size = 32
    topk = 7



@register_task('mnli')
class MNLIConfig(BaseConfig):
    task = 'mnli'
    task_type = 'classification'
    test_subset = 'validation_matched'
    dataset_name = ['glue', 'mnli']
    metrics = 'accuracy'
    templates = [
        'premise|?|<mask>|,|hypothesis|.|',
    ]
    labels = [2, 0, 1]
    remove_columns = ['premise', 'hypothesis', 'idx']
    label_mappings = [('No', 'Yes', 'Maybe'),]
    # TODO: should design a feature to only delete punc for certain columns
    remove_punc = True




@register_task('qnli')
class SNLIConfig(BaseConfig):
    task = 'glue', 'qnli'
    task_type = 'classification'
    dataset_name = 'snli'
    metrics = 'accuracy'
    templates = [
        'premise|.|<mask>|,|hypothesis|.|',
        'premise|!|<mask>|,|hypothesis|.|',
        'premise|.|<mask>|I think,|hypothesis|.|',
        'premise|.|I think|<mask>|,|hypothesis|.|',
        'premise|in the background. |<mask>|,|I think,|hypothesis|.|',
    ]
    labels = [2, 0, 1]
    remove_columns = ['premise', 'hypothesis']
    label_mappings = [('Instead', 'Indeed', 'Seriously'),
                     ('Normally', 'YES', 'This'),
                     ('Meanwhile', 'Right', 'Specifically'),
                     ('Unless', 'Regardless', 'Fortunately'),
                     ('Otherwise', 'Ok', 'Specifically'),
                     ('Except', 'Alright', 'Watch'),
                     ('Plus', 'Sometimes', 'Hopefully'),]
    remove_punc = True
#
# ('Instead', 'Indeed', 'Seriously'),
#  ('Instead', 'Indeed', 'Next'),
#  ('Normally', 'YES', 'This'),
#  ('Meanwhile', 'Again', 'Basically'),
#  ('Meanwhile', 'Right', 'Specifically'),
#  ('Unless', 'Regardless', 'Fortunately'),
#  ('Instead', 'OK', 'Today'),
#  ('Otherwise', 'Ok', 'Specifically'),
#  ('Likewise', 'YES', 'Except'),
#  ('Otherwise', 'YES', 'Plus'),
#  ('Except', 'Alright', 'Watch'),
#  ('Otherwise', 'Still', 'Typically'),
#  ('Later', 'YES', 'This'),
#  ('Unless', 'Exactly', 'Watch'),
#  ('Meanwhile', 'Again', 'Remember'),
#  ('Plus', 'Sometimes', 'Hopefully'),
#  ('Also', 'Yeah', 'Apparently'),
#  ('Oh', 'Okay', 'Certainly'),


@register_task('sst2_demo')
class SST2Config(BaseConfig):
    # Required: The unique task identifier for this task
    task = 'sst2_demo'
    # Required: the task type, each task type corresponds to a data processor
    task_type = 'classification_demo'
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
        'sentence|What a |<mask>| movie.'
    ]
    # Required: The label for this task
    labels = [0, 1]
    # Required: Manually-designed label for each class, the order of `labels` and label_mapping` should match
    label_mappings = [
    ['bad', 'terrible', 'awful', 'dire', 'dread', 'fearful', 'horrible', 'imposing', 'shocking', 'solemn',],
    ['great', 'good', 'right', 'sound', 'pious', 'competent', 'righteous', 'proper', 'upright', 'excellent']
    ]
    # Optional: choose the majority class of highest-topk label candidates
    topk = 7



@register_task('sst2_qa')
class SST2Config(BaseConfig):
    # Required: The unique task identifier for this task
    task = 'sst2_qa'
    # Required: the task type, each task type corresponds to a data processor
    task_type = 'classification_demo'
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
        'sentence|It was?'
    ]
    # Required: The label for this task
    labels = [0, 1]
    # Required: Manually-designed label for each class, the order of `labels` and label_mapping` should match
    label_mappings = [
    ['bad', 'terrible', 'awful', 'dire', 'dread', 'dreadful', 'fearful', 'grand', 'horrible', 'imposing', 'majestic', 'shocking', 'solemn',],
    ['great', 'good', 'right', 'sound', 'pious', 'benevolent', 'competent', 'real', 'considerable', 'righteous', 'proper', 'upright', 'excellent']
    ]
    # Optional: choose the majority class of highest-topk label candidates
    topk = 7
    qa_prompting = True