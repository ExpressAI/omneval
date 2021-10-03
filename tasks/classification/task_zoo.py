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
    remove_columns = ['sentence', 'idx']
    label_mappings = [('terrible', 'great')]


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