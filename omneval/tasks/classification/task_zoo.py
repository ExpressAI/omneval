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
        'sentence|This is |<mask>|.',
        'This is |<mask>|.|sentence|',
        'sentence|A |<mask>| movie.',
        'A |<mask>| movie.|sentence|',
        'sentence|<mask>|!',
        '<mask>|,|sentence|',
        'The author of the following review expresses a |<mask>| sentiment.|sentence|',
        'sentence|The author of the above review expresses a |<mask>| sentiment.'
    ]
    # # Required: The label for this task
    labels = [0, 1]
    # # Required: Manually-designed label for each class, the order of `labels` and label_mapping` should match
    label_mappings = [
    [
        ['bad', 'terrible', 'awful', 'dire', 'horrible', 'abnormal', 'shocking', 'negative', 'rubbish', 'poor'],
        ['great', 'good', 'right', 'sound', 'adorable', 'noble', 'pleasant', 'decent', 'excellent',
         'positive']
    ]
    ]
    # alternative: proper
    # Optional: choose the majority class of highest-topk label candidates
    # Optional: choose the appropriate inference settings
    eval_batch_size = 32
    max_seq_length = 128


@register_task('imdb')
class IMDBConfig(BaseConfig):
    task = 'imdb'
    task_type = 'classification'
    dataset_name = ['imdb']
    metrics = 'accuracy'
    test_subset = 'test'
    templates = [
        'text|It was |<mask>|.',
        'It was |<mask>|.|text|',
        'text|This is |<mask>|.',
        'This is |<mask>|.|text|',
        'text|A |<mask>| movie.',
        'A |<mask>| movie.|text|',
        'text|<mask>|!',
        '<mask>|,|text|',
        'The author of the following review expresses a |<mask>| sentiment.|text|',
        'text|The author of the above review expresses a |<mask>| sentiment.'
    ]
    labels = [0, 1]
    label_mappings = [
    [
        ['bad', 'terrible', 'awful', 'dire', 'horrible', 'abnormal', 'shocking', 'negative', 'rubbish', 'poor'],
        ['great', 'good', 'right', 'sound', 'adorable', 'noble', 'pleasant', 'decent', 'excellent',
         'positive']
    ]
    ]
    eval_batch_size = 8
    max_seq_length = 512

@register_task('rotten_tomatoes')
class RTConfig(BaseConfig):
    task = 'rotten_tomatoes'
    task_type = 'classification'
    dataset_name = ['rotten_tomatoes']
    metrics = 'accuracy'
    test_subset = 'test'
    templates = [
        'text|It was |<mask>|.',
        'It was |<mask>|.|text|',
        'text|This is |<mask>|.',
        'This is |<mask>|.|text|',
        'text|A |<mask>| movie.',
        'A |<mask>| movie.|text|',
        'text|<mask>|!',
        '<mask>|,|text|',
        'The author of the following review expresses a |<mask>| sentiment.|text|',
        'text|The author of the above review expresses a |<mask>| sentiment.'
    ]
    labels = [0, 1]
    label_mappings = [
    [
        ['bad', 'terrible', 'awful', 'dire', 'horrible', 'abnormal', 'shocking', 'negative', 'rubbish', 'poor'],
        ['great', 'good', 'right', 'sound', 'adorable', 'noble', 'pleasant', 'decent', 'excellent',
         'positive']
    ]
    ]
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
        'The following two sentences are |<mask>|.|premise|.|hypothesis|.|',
        'premise|.|hypothesis|.|The above two sentences are |<mask>|.',
        'Because |premise|, |hypothesis| is |<mask>|.',
        'It is |<mask>| that |hypothesis|, because |premise|.'
    ]
    labels = [2, 0, 1]
    templates_answers_mapping = [0, 0, 0, 1, 1, 2, 2]
    label_mappings = [
    [
        ['No','Instead', 'But', 'Otherwise', 'Yet', 'Except', 'However', 'Rather'],
        ['Yes', 'Exactly', 'Right', 'Absolutely', 'Yeah', 'Therefore', 'Definitely', 'Indeed'],
        ['Maybe', 'Probably', 'Perhaps', 'And', 'Possibly', 'Likely', 'Also', 'Or']
    ],
    [
        ['opposite', 'different', 'opposed', 'counter', 'anti', 'against'],
        ['associated', 'linked', 'related', 'equal', 'similar', 'like'],
        ['detached', 'irrelevant', 'independent', 'separate', 'apart', 'divided']
    ],
    [
        ['false', 'wrong', 'flawed', 'misleading', 'fake', 'invalid', 'inaccurate'],
        ['true', 'exact', 'right', 'correct', 'real', 'precise', 'valid'],
        ['possible', 'confusing', 'unknown', 'hidden', 'concealed', 'secret', 'open']
    ],
    ]
    # Optional: choose the majority class of highest-topk label candidates
    remove_punc = True

    eval_batch_size = 32
    max_seq_length = 128


@register_task('snli')
class SNLIConfig(BaseConfig):
    task = 'snli'
    task_type = 'classification'
    dataset_name = ['snli']  # datasets.load_dataset('glue', 'sst2')
    metrics = 'accuracy'
    test_subset = 'validation'
    templates = [
        'premise|?|<mask>|,|hypothesis|.|',
        'premise|,|<mask>|,|hypothesis|.|',
        'premise|!|<mask>|,|hypothesis|.|',
        'The following two sentences are |<mask>|.|premise|.|hypothesis|.|',
        'premise|.|hypothesis|.|The above two sentences are |<mask>|.',
        'Because |premise|, |hypothesis| is |<mask>|.',
        'It is |<mask>| that |hypothesis|, because |premise|.'
    ]
    labels = [2, 0, 1]
    templates_answers_mapping = [0, 0, 0, 1, 1, 2, 2]
    label_mappings = [
    [
        ['No','Instead', 'But', 'Otherwise', 'Yet', 'Except', 'However', 'Rather'],
        ['Yes', 'Exactly', 'Right', 'Absolutely', 'Yeah', 'Therefore', 'Definitely', 'Indeed'],
        ['Maybe', 'Probably', 'Perhaps', 'And', 'Possibly', 'Likely', 'Also', 'Or']
    ],
    [
        ['opposite', 'different', 'opposed', 'counter', 'anti', 'against'],
        ['associated', 'linked', 'related', 'equal', 'similar', 'like'],
        ['detached', 'irrelevant', 'independent', 'separate', 'apart', 'divided']
    ],
    [
        ['false', 'wrong', 'flawed', 'misleading', 'fake', 'invalid', 'inaccurate'],
        ['true', 'exact', 'right', 'correct', 'real', 'precise', 'valid'],
        ['possible', 'confusing', 'unknown', 'hidden', 'concealed', 'secret', 'open']
    ],
    ]
    remove_punc = True
    eval_batch_size = 32
    max_seq_length = 128


@register_task('rte')
class RTEConfig(BaseConfig):
    task = 'rte'
    task_type = 'classification'
    dataset_name = ['glue', 'rte']
    test_subset = 'validation'
    metrics = 'accuracy'
    templates = [
        'sentence1|?|<mask>|,|sentence2|.|',
        'sentence1|,|<mask>|,|sentence2|.|',
        'sentence1|!|<mask>|,|sentence2|.|',
        'The following two sentences are |<mask>|.|sentence1|.|sentence2|.|',
        '|sentence1|.|sentence2|.|The above two sentences are |<mask>|.',  # TODO: seems this templates is wrong in the experiment
        'Because |sentence1|, |sentence2| is |<mask>|.',
        'It is |<mask>| that |sentence2|, because |sentence1|.'
    ]
    labels = [0, 1]
    templates_answers_mapping = [0, 0, 0, 1, 1, 2, 2]
    label_mappings = [
    [
        ['No','Instead', 'But', 'Otherwise', 'Yet', 'Except', 'However', 'Rather'],
        ['Yes', 'Exactly', 'Right', 'Absolutely', 'Yeah', 'Therefore', 'Definitely', 'Indeed'],
    ],
    [
        ['opposite', 'different', 'opposed', 'counter', 'anti', 'against'],
        ['associated', 'linked', 'related', 'equal', 'similar', 'like'],
    ],
    [
        ['true', 'exact', 'right', 'correct', 'real', 'precise', 'valid'],
        ['false', 'wrong', 'flawed', 'misleading', 'fake', 'invalid', 'inaccurate'],
    ],
    ]
    remove_punc = True
    eval_batch_size = 32
    max_seq_length = 128


@register_task('mrpc')
class MRPCConfig(BaseConfig):
    task = 'mrpc'
    task_type = 'classification'
    dataset_name = ['glue', 'mrpc']
    test_subset = 'test'
    metrics = 'f1'
    templates = [
        'sentence1|<mask>|,|sentence2',
        'The following two sentences are |<mask>|.|sentence1|sentence2',
        'sentence1|sentence2|The above two sentences are |<mask>|.',
    ]
    labels = [0, 1]
    templates_answers_mapping = [0, 1, 1]
    label_mappings = [
    [
        ['No','Instead', 'But', 'Otherwise', 'Yet', 'Except', 'However', 'Rather'],
        ['Yes', 'Exactly', 'Right', 'Absolutely', 'Yeah', 'Therefore', 'Definitely', 'Indeed'],
    ],
    [
        ['detached', 'irrelevant', 'independent', 'separate', 'apart', 'divided', 'different'],
        ['associated', 'linked', 'related', 'equal', 'similar', 'like', 'same'],
    ],
    ]
    eval_batch_size = 32
    max_seq_length = 128
    remove_punc = True


@register_task('qqp')
class QQPConfig(BaseConfig):
    task = 'qqp'
    task_type = 'classification'
    dataset_name = ['glue', 'qqp']
    test_subset = 'validation'
    metrics = 'f1'
    templates = [
     'question1|<mask>|,|question2',
     'The following two questions are |<mask>|.|question1|question2',
     'question1|question2|The above two questions are |<mask>|.',
    ]
    labels = [0, 1]
    templates_answers_mapping = [0, 1, 1]
    label_mappings = [
    [
        ['No', 'Instead', 'But', 'Otherwise', 'Yet', 'Except', 'However', 'Rather'],
        ['Yes', 'And', 'Likely', 'Similarly', 'Also', 'Again', 'Yeah', 'Right'],

    ],
    [
        ['detached', 'irrelevant', 'independent', 'separate', 'apart', 'divided', 'different'],
        ['associated', 'linked', 'related', 'equal', 'similar', 'like', 'same'],
    ],
    ]
    eval_batch_size = 32
    max_seq_length = 128


@register_task('ag_news')
class AgConfig(BaseConfig):
    task = 'ag_news'
    task_type = 'classification'
    dataset_name = ['ag_news']
    metrics = 'accuracy'
    test_subset = 'test'
    templates = [
        'text|It is about |<mask>|.',
        'It is about |<mask>|.|text|',
        'text|A piece of |<mask>| news.',
        'A piece of |<mask>| news.|text|',
        'text|<mask>|!',
        '<mask>|,|text|',
        'The topic of the following news is |<mask>|.|text|',
        'text|The topic of the above news is |<mask>|.'
    ]
    labels = [0, 1, 2, 3]
    label_mappings = [
    [
        ['world', 'politics', 'government', 'nations'],
        ['sports', 'health', 'tournament', 'games'],
        ['business', 'finance', 'money', 'trade'],
        ['science', 'tech', 'engineer', 'design']
    ]
    ]
    eval_batch_size = 16
    max_seq_length = 256


@register_task('dbpedia_14')
class DBPediaConfig(BaseConfig):
    task = 'dbpedia_14'
    task_type = 'classification'
    dataset_name = ['ag_news']
    metrics = 'accuracy'
    test_subset = 'test'
    templates = [
        'text|It is about |<mask>|.',
        'It is about |<mask>|.|text|',
        'text|A |<mask>| article.',
        'A |<mask>| article.|text|',
        'text|<mask>|!',
        '<mask>|,|text|',
        'The topic of the following article is |<mask>|.|text|',
        'text|The topic of the above article is |<mask>|.'
    ]
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    label_mappings = [
    [
        ['company'],
        ['education'],
        ['artist'],
        ['athlete'],
        ['officer'],
        ['transportation'],
        ['building'],
        ['nature'],
        ['village'],
        ['animal'],
        ['plant'],
        ['album'],
        ['film'],
        ['text']
    ]
    ]
    eval_batch_size = 16
    max_seq_length = 256
