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

@register_task('sst2_m')
class SST2Config(BaseConfig):
    task = 'sst2_m'
    task_type = 'classification_m'
    dataset_name = ['glue', 'sst2']
    test_subset = 'validation'
    metrics = 'accuracy'
    templates = [
        'sentence|It was |<mask>|.'
    ]
    labels = [0, 1]
    remove_columns = ['sentence', 'idx']
    label_mappings = [
    ['bad', 'terrible', 'awful', 'dire', 'dread', 'dreadful', 'fearful', 'grand', 'horrible', 'imposing', 'majestic', 'shocking', 'solemn',],
    ['great', 'good', 'right', 'sound', 'pious', 'benevolent', 'competent', 'real', 'considerable', 'righteous', 'proper', 'upright', 'excellent']
    ]
    topk = 7



@register_task('sst2_qa')
class SST2Config(BaseConfig):
    task = 'sst2_qa'
    task_type = 'classification'
    dataset_name = ['glue', 'sst2']
    test_subset = 'validation'
    metrics = 'accuracy'
    templates = [
        'sentence|It was ?'
    ]
    labels = [0, 1]
    remove_columns = ['sentence', 'idx']
    label_mappings = [('terrible', 'great')]
    qa_prompting = True


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