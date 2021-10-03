from .. import BaseConfig, register_task


@register_task('sst2')
class SST2Config(BaseConfig):
    task = 'sst2'
    task_type = 'classification'
    dataset_name = 'sst'
    metrics = 'accuracy'
    templates = [
        'sentence|It was |<mask>|.',
        'sentence|This is |<mask>|.',
        'sentence|A |<mask>| film.',
        'It was |<mask>|.|sentence',
        'This is |<mask>|.|sentence',
        'A |<mask>| film.|sentence',
    ]
    labels = [0, 1]
    remove_columns = ['sentence', 'tokens', 'tree']
    label_mappings = [('unnecessary', 'inspiring'),
                    ('ridiculous', 'flawless'),
                    ('embarrassing', 'marvelous'),
                    ('stupid', 'timely'),
                    ('nothing', 'successful'),
                    ('dreadful', 'exquisite'),
                    ('pointless', 'fabulous'),
                    ('horrible', 'incredible'),
                    ('worse', 'memorable'),
                    ('awkward', 'inspiring'),
                    ('obvious', 'timely'),
                    ('disgusting', 'thrilling'),
                    ('dreadful', 'marvelous'),
                    ('weird', 'exciting'),
                    ('disgusting', 'marvelous'),
                    ('awkward', 'spectacular'),
                    ('disastrous', 'exceptional'),
                    ('bizarre', 'sublime'),
                    ('boring', 'remarkable'),
                    ('horrible', 'epic'),
                    ('ugly', 'timely'),
                    ('disgusting', 'irresistible'),
                    ('ridiculous', 'exquisite'),
                    ('pointless', 'astonishing'),
                    ('unnecessary', 'sublime'),
                    ('stupid', 'fabulous'),
                    ('worse', 'superb'),
                    ('annoying', 'exquisite'),
                    ('abandoned', 'exquisite'),
                    ('obvious', 'profound'),
                    ('worse', 'charming'),
                    ('frustrating', 'irresistible'),
                    ('ridiculous', 'gorgeous'),
                    ('awkward', 'magical'),
                    ('horrible', 'spectacular'),
                    ('embarrassing', 'irresistible'),
                    ('boring', 'perfection'),
                    ('weird', 'inspiring'),
                    ('sad', 'inspiring'),
                    ('ridiculous', 'thrilling'),
                    ('pathetic', 'fabulous'),
                    ('horrible', 'delicious'),
                    ('unnecessary', 'magical'),
                    ('pointless', 'spectacular'),
                    ('pointless', 'exceptional'),
                    ('ugly', 'profound'),
                    ('embarrassing', 'delightful'),
                    ('obvious', 'inspiring'),
                    ('embarrassing', 'exquisite'),
                    ('embarrassing', 'flawless'),
                    ('ridiculous', 'magnificent'),
                    ('ridiculous', 'delightful'),
                    ('boring', 'fabulous'),
                    ('disgusting', 'exquisite'),
                    ('worse', 'extraordinary')]


@register_task('mnli')
class MNLIConfig(BaseConfig):
    task = 'mnli'
    task_type = 'classification'
    dataset_name = 'multi_nli'
    metrics = 'accuracy'
    templates = [
        'sentence|It was |<mask>|.',
        'sentence|This is |<mask>|.',
        'sentence|A |<mask>| film.',
        'It was |<mask>|.|sentence',
        'This is |<mask>|.|sentence',
        'A |<mask>| film.|sentence',
    ]
    labels = [0, 1]
    remove_columns = ['sentence', 'tokens', 'tree']
    label_mappings = [('Still', 'Basically', 'And'),
                     ('Otherwise', 'Then', 'Plus'),
                     ('Personally', 'Exactly', 'Probably'),
                     ('Next', 'Exactly', 'indeed'),
                     ('no', 'yes', 'Nonetheless'),
                     ('Nah', 'Literally', 'Why'),
                     ('well', 'Seriously', 'Fortunately'),
                     ('But', 'Yeah', 'Clearly'),
                     ('Instead', 'Specifically', 'Therefore'),
                     ('Meanwhile', 'Right', 'Probably'),]



@register_task('snli')
class SNLIConfig(BaseConfig):
    task = 'snli'
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