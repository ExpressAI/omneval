from omneval.tasks import BaseConfig
from omneval.registry import register_task


# TODO: need to relate this part with Datalabs
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
    topk = 1
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


@register_task('yelp_polarity')
class AmazonPopularityConfig(BaseConfig):
    task = 'yelp_polarity'
    task_type = 'classification'
    dataset_name = ['yelp_polarity']
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
        'text|The author of the above review expresses a |<mask>| sentiment.',
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
    max_seq_length = 256

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
    templates_answers_mapping = [0, 0, 0, 0, 0, 0, 1, 1, 2, 2]
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
    dataset_name = ['snli']
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
        ['Yes', 'Exactly', 'Right', 'Absolutely', 'Yeah', 'Therefore', 'Definitely', 'Indeed'],
        ['No','Instead', 'But', 'Otherwise', 'Yet', 'Except', 'However', 'Rather'],
    ],
    [
        ['associated', 'linked', 'related', 'equal', 'similar', 'like'],
        ['opposite', 'different', 'opposed', 'counter', 'anti', 'against'],
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
    metrics = 'accuracy'
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


@register_task('qqp')
class QQPConfig(BaseConfig):
    task = 'qqp'
    task_type = 'classification'
    dataset_name = ['glue', 'qqp']
    test_subset = 'validation'
    metrics = 'accuracy'
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
    dataset_name = ['dbpedia_14']
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


@register_task('amazon_polarity')
class AmazonPopularityConfig(BaseConfig):
    task = 'amazon_polarity'
    task_type = 'classification'
    dataset_name = ['amazon_polarity']
    metrics = 'accuracy'
    test_subset = 'test'
    templates = [
        'title|.|content|It was |<mask>|.',
        'It was |<mask>|.|title|.|content|',
        'title|.|content|This is |<mask>|.',
        'This is |<mask>|.|title|.|content|',
        'title|.|content|<mask>|!',
        '<mask>|,|title|.|content|',
        'The author of the following review expresses a |<mask>| sentiment.|title|.|content|',
        'title|.|content|The author of the above review expresses a |<mask>| sentiment.'
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
    max_seq_length = 256


@register_task('sem_eval_2014_task_1')
class SNLIConfig(BaseConfig):
    task = 'sem_eval_2014_task_1'
    task_type = 'classification'
    dataset_name = ['sem_eval_2014_task_1']  # datasets.load_dataset('glue', 'sst2')
    metrics = 'accuracy'
    test_subset = 'test'
    label_name = 'entailment_judgment'
    templates = [
        'premise|?|<mask>|,|hypothesis|.|',
        'premise|,|<mask>|,|hypothesis|.|',
        'premise|!|<mask>|,|hypothesis|.|',
        'The following two sentences are |<mask>|.|premise|.|hypothesis|.|',
        'premise|.|hypothesis|.|The above two sentences are |<mask>|.',
        'Because |premise|, |hypothesis| is |<mask>|.',
        'It is |<mask>| that |hypothesis|, because |premise|.'
    ]
    labels = [2, 1, 0]
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


@register_task('qnli')
class RTEConfig(BaseConfig):
    task = 'qnli'
    task_type = 'classification'
    dataset_name = ['glue', 'qnli']
    test_subset = 'validation'
    metrics = 'accuracy'
    templates = [
        'question|<mask>|,|sentence|',
        'The following two sentences are |<mask>|.|question|sentence',
        'question|sentence|The above two sentences are |<mask>|.',  # TODO: seems this templates is wrong in the experiment
    ]
    labels = [0, 1]
    templates_answers_mapping = [0, 1, 1]
    label_mappings = [
    [
        ['Yes', 'Exactly', 'Right', 'Absolutely', 'Yeah', 'Therefore', 'Definitely', 'Indeed'],
        ['No','Instead', 'But', 'Otherwise', 'Yet', 'Except', 'However', 'Rather'],
    ],
    [
        ['associated', 'linked', 'related', 'equal', 'similar', 'like'],
        ['opposite', 'different', 'opposed', 'counter', 'anti', 'against'],
    ],
    [
        ['true', 'exact', 'right', 'correct', 'real', 'precise', 'valid'],
        ['false', 'wrong', 'flawed', 'misleading', 'fake', 'invalid', 'inaccurate'],
    ],
    ]
    eval_batch_size = 32
    max_seq_length = 256


@register_task('trec')
class TrecConfig(BaseConfig):
    task = 'trec'
    task_type = 'classification'
    dataset_name = ['trec']
    metrics = 'accuracy'
    test_subset = 'test'
    templates = [
        'text|It is about |<mask>|.',
        'It is about |<mask>|.|text|',
        'text|A question of |<mask>|.',
        'A question of |<mask>|.|text|',
        'text|<mask>|!',
        '<mask>|,|text|',
        'The topic of the following question is |<mask>|.|text|',
        'text|The topic of the above question is |<mask>|.'
    ]
    label_name = 'label-coarse'
    labels = [0, 1, 2, 3, 4, 5]
    label_mappings = [
        [
            ['contraction', 'compression', 'symbol'],
            ['description', 'explanation', 'representation'],
            ['entities', 'object', 'substance'],
            ['human', 'people', 'person'],
            ['locations', 'place', 'position'],
            ['numbers', 'quantity', 'figure']
        ]
    ]
    eval_batch_size = 32
    max_seq_length = 128


@register_task('paws')
class QQPConfig(BaseConfig):
    task = 'paws'
    task_type = 'classification'
    dataset_name = ['paws', 'labeled_final']
    test_subset = 'test'
    metrics = 'accuracy'
    templates = [
     'sentence1|<mask>|,|sentence2',
     'The following two sentences are |<mask>|.|sentence1|sentence2',
     'sentence1|sentence2|The above two sentences are |<mask>|.',
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


@register_task('wnli')
class RTEConfig(BaseConfig):
    task = 'wnli'
    task_type = 'classification'
    dataset_name = ['glue', 'wnli']
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
    labels = [1, 0]
    templates_answers_mapping = [0, 0, 0, 1, 1, 2, 2]
    label_mappings = [
    [
        ['Yes', 'Exactly', 'Right', 'Absolutely', 'Yeah', 'Therefore', 'Definitely', 'Indeed'],
        ['No','Instead', 'But', 'Otherwise', 'Yet', 'Except', 'However', 'Rather'],
    ],
    [
        ['associated', 'linked', 'related', 'equal', 'similar', 'like'],
        ['opposite', 'different', 'opposed', 'counter', 'anti', 'against'],
    ],
    [
        ['true', 'exact', 'right', 'correct', 'real', 'precise', 'valid'],
        ['false', 'wrong', 'flawed', 'misleading', 'fake', 'invalid', 'inaccurate'],
    ],
    ]
    remove_punc = True
    eval_batch_size = 32
    max_seq_length = 128


@register_task('medical_questions_pairs')
class MQPConfig(BaseConfig):
    task = 'medical_questions_pairs'
    task_type = 'classification'
    dataset_name = ['medical_questions_pairs']
    test_subset = 'train'
    metrics = 'accuracy'
    templates = [
     'question_1|<mask>|,|question_2',
     'The following two questions are |<mask>|.|question_1|question_2',
     'question_1|question_2|The above two questions are |<mask>|.',
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
    eval_batch_size = 16
    max_seq_length = 256


@register_task('boolq')
class BoolqConfig(BaseConfig):
    task = 'boolq'
    task_type = 'classification'
    dataset_name = ['super_glue', 'boolq']
    test_subset = 'validation'
    metrics = 'accuracy'
    templates = [
     'passage|.|question|?|<mask>|.',
     'question|?|<mask>|.|passage|.',
     'passage|.|Question: |question|?|Answer: |<mask>|.',
     'Question: |question|?|Answer: |<mask>|.|passage|.|',
     'passage|.|Based on the previous passage, |question|?|Answer: |<mask>|.',
     'Based on the following passage, |question|?|Answer: |<mask>|.|passage|.',
    ]
    labels = [1, 0]
    label_mappings = [
    [
        ['yes', 'true', 'exact', 'right', 'correct', 'real', 'precise', 'valid'],
        ['no', 'false', 'wrong', 'flawed', 'misleading', 'fake', 'invalid', 'inaccurate'],
    ],
    ]
    eval_batch_size = 16
    max_seq_length = 256
    remove_punc = True



@register_task('mc_taco')
class BoolqConfig(BaseConfig):
    task = 'mc_taco'
    task_type = 'classification'
    dataset_name = ['mc_taco']
    test_subset = 'test'
    metrics = 'accuracy'
    templates = [
     'sentence|question|answer|.|<mask>|.',
     'sentence|question|A |<mask>| answer is |answer|.',
     'sentence|Question: |question|Answer: |answer|.|This answer is |<mask>|.',
     'Question: |question|Answer: |answer|.|sentence|This answer is |<mask>|.',
     'sentence|Based on the previous sentence, |question|A |<mask>| answer is |answer|.',
     'Based on the following sentence, |question|A |<mask>| answer is |answer|.|sentence|.',
    ]
    labels = [1, 0]
    label_mappings = [
    [
        ['yes', 'true', 'exact', 'right', 'correct', 'real', 'precise', 'valid'],
        ['no', 'false', 'wrong', 'flawed', 'misleading', 'fake', 'invalid', 'inaccurate'],
    ],
    ]
    eval_batch_size = 16
    max_seq_length = 256


def processing_asba(config, data):
    def process(example):
        example['aspect'], example['sentence'], example['label'] = example['text'].split('\t')
        example['label'] = 0 if example['label'] == 'negative' else 1
        return example
    return data.map(process, remove_columns=['text'])


@register_task('absa-twitter')
class ABSATwitterConfig(BaseConfig):
    task = 'absa-twitter'
    task_type = 'classification'
    dataset_name = 'datasets/absa/test-twitter.tsv'
    metrics = 'accuracy'
    test_subset = 'test'
    templates = [
        'sentence|aspect|was|<mask>|.',
        'aspect|was|<mask>|.|sentence|',
        'sentence|The sentiment of |aspect|is|<mask>|.',
        'The sentiment of |aspect|is|<mask>|.|sentence|',
        'sentence|aspect|,|<mask>|!',
        'aspect|:|<mask>|,|sentence',
        'The author of the following review expresses a |<mask>| sentiment on |aspect|.|sentence',
        'sentence|The author of the above review expresses a |<mask>| sentiment on |aspect|.'
    ]
    labels = [0, 1]
    label_mappings = [
    [
        ['bad', 'terrible', 'awful', 'dire', 'horrible', 'abnormal', 'shocking', 'negative', 'rubbish', 'poor'],
        ['great', 'good', 'right', 'sound', 'adorable', 'noble', 'pleasant', 'decent', 'excellent',
         'positive']
    ]
    ]
    eval_batch_size = 16
    max_seq_length = 128
    data_preprocessing = processing_asba


@register_task('absa-laptop')
class ABSALaptopConfig(BaseConfig):
    task = 'absa-laptop'
    task_type = 'classification'
    dataset_name = 'datasets/absa/test-laptop.tsv'
    metrics = 'accuracy'
    test_subset = 'test'
    templates = [
        'sentence|aspect|was|<mask>|.',
        'aspect|was|<mask>|.|sentence|',
        'sentence|The sentiment of |aspect|is|<mask>|.',
        'The sentiment of |aspect|is|<mask>|.|sentence|',
        'sentence|aspect|,|<mask>|!',
        'aspect|:|<mask>|,|sentence',
        'The author of the following review expresses a |<mask>| sentiment on |aspect|.|sentence',
        'sentence|The author of the above review expresses a |<mask>| sentiment on |aspect|.'
    ]
    labels = [0, 1]
    label_mappings = [
    [
        ['bad', 'terrible', 'awful', 'dire', 'horrible', 'abnormal', 'shocking', 'negative', 'rubbish', 'poor'],
        ['great', 'good', 'right', 'sound', 'adorable', 'noble', 'pleasant', 'decent', 'excellent',
         'positive']
    ]
    ]
    eval_batch_size = 16
    max_seq_length = 128
    data_preprocessing = processing_asba


@register_task('absa-rest14')
class ABSARest14Config(BaseConfig):
    task = 'absa-rest14'
    task_type = 'classification'
    dataset_name = 'datasets/absa/test-rest14.tsv'
    metrics = 'accuracy'
    test_subset = 'test'
    templates = [
        'sentence|aspect|was|<mask>|.',
        'aspect|was|<mask>|.|sentence|',
        'sentence|The sentiment of |aspect|is|<mask>|.',
        'The sentiment of |aspect|is|<mask>|.|sentence|',
        'sentence|aspect|,|<mask>|!',
        'aspect|:|<mask>|,|sentence',
        'The author of the following review expresses a |<mask>| sentiment on |aspect|.|sentence',
        'sentence|The author of the above review expresses a |<mask>| sentiment on |aspect|.'
    ]
    labels = [0, 1]
    label_mappings = [
    [
        ['bad', 'terrible', 'awful', 'dire', 'horrible', 'abnormal', 'shocking', 'negative', 'rubbish', 'poor'],
        ['great', 'good', 'right', 'sound', 'adorable', 'noble', 'pleasant', 'decent', 'excellent',
         'positive']
    ]
    ]
    eval_batch_size = 16
    max_seq_length = 128
    data_preprocessing = processing_asba