from omneval.tasks import BaseConfig
from omneval.registry import register_task
from omneval.utils import make_sentence, get_entity_span_ids
import pandas as pd
from datasets import Dataset
import pdb
import logging
try:
    from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
except:
    logging.info('allennlp not installed, staructural prediction tasks disabled')


def processing_conll2003(config, data):
    examples = []
    for idx, item in enumerate(data):
        # TODO: Do we need to normalize each vocabulary of the text?
        # Only to test
        sentence = ' '.join(item['tokens'])
        span_ids = set(enumerate_spans(item['tokens'], max_span_width=config.max_span_width))
        entity_span_ids = get_entity_span_ids(item['ner_tags'], config.tags)
        for label, start_id, end_id in entity_span_ids:
            span = item['tokens'][start_id: end_id+1]
            examples.append({'sentence_id': idx, 'span': span, 'label': label, 'sentence': sentence,
                             'span_idx': [start_id, end_id+1]})
            span_ids.discard((start_id, end_id))
        for start_id, end_id in span_ids:
            span = item['tokens'][start_id: end_id+1]
            examples.append({'sentence_id': idx, 'span': span, 'label': 'O', 'sentence': sentence,
                             'span_idx': [start_id, end_id+1]})
    return Dataset.from_pandas(pd.DataFrame(examples))


def processing_conll2000(config, data):
    examples = []
    len_num_dic = {}
    for idx, item in enumerate(data):
        # TODO: Do we need to normalize each vocabulary of the text?
        # Only to test
        sentence = ' '.join(item['tokens'])
        span_ids = set(enumerate_spans(item['tokens'], max_span_width=config.max_span_width))
        entity_span_ids = get_entity_span_ids(item['chunk_tags'], config.tags)
        for label, start_id, end_id in entity_span_ids:
            span = item['tokens'][start_id: end_id+1]
            examples.append({'sentence_id': idx, 'span': span, 'label': label, 'sentence': sentence,
                             'span_idx': [start_id, end_id+1]})
            span_ids.discard((start_id, end_id))

        for start_id, end_id in span_ids:
            span = item['tokens'][start_id: end_id+1]
            examples.append({'sentence_id': idx, 'span': span, 'label': 'O', 'sentence': sentence,
                             'span_idx': [start_id, end_id+1]})
    return Dataset.from_pandas(pd.DataFrame(examples))


def processing_conll2003_chunk(config, data):
    examples = []
    for idx, item in enumerate(data):
        # TODO: Do we need to normalize each vocabulary of the text?
        # Only to test
        sentence = ' '.join(item['tokens'])
        span_ids = set(enumerate_spans(item['tokens'], max_span_width=config.max_span_width))
        entity_span_ids = get_entity_span_ids(item['chunk_tags'], config.tags)
        for label, start_id, end_id in entity_span_ids:
            span = item['tokens'][start_id: end_id+1]
            examples.append({'sentence_id': idx, 'span': span, 'label': label, 'sentence': sentence,
                             'span_idx': [start_id, end_id+1]})
            span_ids.discard((start_id, end_id))
        for start_id, end_id in span_ids:
            span = item['tokens'][start_id: end_id+1]
            examples.append({'sentence_id': idx, 'span': span, 'label': 'O', 'sentence': sentence,
                             'span_idx': [start_id, end_id+1]})
    return Dataset.from_pandas(pd.DataFrame(examples))

def processing_conll2003_pos(config, data):
    examples = []
    for idx, item in enumerate(data):
        # TODO: Do we need to normalize each vocabulary of the text?
        # Only to test
        sentence = ' '.join(item['tokens'])
        tokens = item['tokens']
        tags = item['pos_tags']
        for i,(token,tag) in enumerate(zip(tokens,tags)):
            examples.append({'sentence_id': idx, 'span': token, 'label': tag, 'sentence': sentence,
                             'span_idx': [i, i + 1]})

    return Dataset.from_pandas(pd.DataFrame(examples))


@register_task('conll2003')
class ConllConfig(BaseConfig):
    # Required: The unique task identifier for this task
    task = 'conll2003'
    # Required: the task type, each task type corresponds to a data processor
    task_type = 'structure_prediction'
    # Required: Either input a file name that can be tracked in the environment(like  '${PATH_TO_FILE}/${FILE_NAME}')
    # or a str or list, which is a dataset name for huggingface's `datasets`
    dataset_name = 'conll2003'  # datasets.load_dataset('glue', 'sst2')
    # dataset_name = 'lama.json'  # datasets.load_dataset('json', 'lama.json')
    # dataset_name = 'lama '    # datasets.load_dataset('lama')
    # Required: The metrics used for this task, using metrics in huggingface.Metrics or defined metrics in metrics.py
    metrics = 'f1-ner'
    # metrics_kwargs = {'average': 'micro'}
    # Optional: The data split used for evaluation: default 'test'
    test_subset = 'test'

    # Below are parameters for text classification/structural prediction
    # Required: prompt template:
    # e.g  `sentence` is the column name for the raw text "It was" and "." are templates, <mask> is the masked poition
    # Then the template is "<text> It was <mask>." <e> is for the entity tokens
    # ex: Obama was the president of USA. Obama is [MASK] [MASK] entity. (ans: a person)
    templates = [
        'sentence|<e>|is |<mask>|entity.',
    ]
    # Required: The label for this task
    labels = ['O', 'PER', 'ORG', 'LOC', 'MISC']
    # Required: Manually-designed label for each class, the order of `labels` and label_mapping` should match
    tags = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    label_mappings = [
        [
            ['not an'],
            ['a person'],
            ['an organization'],
            ['a location'],
            ['an other'],
        ]
    ]
    # Optional: specify the name of the label column
    label_name = 'label'
    sentence_label = 'sentence'
    remove_punc = False
    data_preprocessing = processing_conll2003
    max_span_width = 6
    eval_batch_size = 8


def processing_wikiann(config, data):
    examples = []
    for idx, item in enumerate(data):
        # TODO: Do we need to normalize each vocabulary of the text?
        # Only to test
        sentence = ' '.join(item['tokens'])
        span_ids = set(enumerate_spans(item['tokens'], max_span_width=config.max_span_width))
        entity_span_ids = get_entity_span_ids(item['ner_tags'], config.tags)
        for label, start_id, end_id in entity_span_ids:
            span = item['tokens'][start_id: end_id+1]
            examples.append({'sentence_id': idx, 'span': span, 'label': label, 'sentence': sentence,
                             'span_idx': [start_id, end_id+1]})
            span_ids.discard((start_id, end_id))
        for start_id, end_id in span_ids:
            span = item['tokens'][start_id: end_id+1]
            examples.append({'sentence_id': idx, 'span': span, 'label': 'O', 'sentence': sentence,
                             'span_idx': [start_id, end_id+1]})
    return Dataset.from_pandas(pd.DataFrame(examples))

@register_task('wikiann')
class ConllConfig(BaseConfig):
    # Required: The unique task identifier for this task
    task = 'wikiann'
    # Required: the task type, each task type corresponds to a data processor
    task_type = 'structure_prediction'
    # Required: Either input a file name that can be tracked in the environment(like  '${PATH_TO_FILE}/${FILE_NAME}')
    # or a str or list, which is a dataset name for huggingface's `datasets`
    dataset_name = ['wikiann', 'en']  # datasets.load_dataset('glue', 'sst2')
    # dataset_name = 'lama.json'  # datasets.load_dataset('json', 'lama.json')
    # dataset_name = 'lama '    # datasets.load_dataset('lama')
    # Required: The metrics used for this task, using metrics in huggingface.Metrics or defined metrics in metrics.py
    metrics = 'f1-ner'
    # metrics_kwargs = {'average': 'micro'}
    # Optional: The data split used for evaluation: default 'test'
    test_subset = 'test'

    # Below are parameters for text classification/structural prediction
    # Required: prompt template:
    # e.g  `sentence` is the column name for the raw text "It was" and "." are templates, <mask> is the masked poition
    # Then the template is "<text> It was <mask>." <e> is for the entity tokens
    # ex: Obama was the president of USA. Obama is [MASK] [MASK] entity. (ans: a person)
    templates = [
        'sentence|<e>|is |<mask>|entity.',
    ]
    # Required: The label for this task
    labels = ['O', 'PER', 'ORG', 'LOC']
    # Required: Manually-designed label for each class, the order of `labels` and label_mapping` should match
    tags = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6}
    label_mappings = [
        [
            ['not an'],
            ['a person'],
            ['an organization'],
            ['a location'],
        ]
    ]
    # Optional: specify the name of the label column
    label_name = 'label'
    sentence_label = 'sentence'
    remove_punc = False
    data_preprocessing = processing_conll2003
    max_span_width = 6


@register_task('wnut_17')
class ConllConfig(BaseConfig):
    # Required: The unique task identifier for this task
    task = 'wnut_17'
    # Required: the task type, each task type corresponds to a data processor
    task_type = 'structure_prediction'
    # Required: Either input a file name that can be tracked in the environment(like  '${PATH_TO_FILE}/${FILE_NAME}')
    # or a str or list, which is a dataset name for huggingface's `datasets`
    dataset_name = 'wnut_17'  # datasets.load_dataset('glue', 'sst2')
    # dataset_name = 'lama.json'  # datasets.load_dataset('json', 'lama.json')
    # dataset_name = 'lama '    # datasets.load_dataset('lama')
    # Required: The metrics used for this task, using metrics in huggingface.Metrics or defined metrics in metrics.py
    metrics = 'f1-ner'
    # metrics_kwargs = {'average': 'micro'}
    # Optional: The data split used for evaluation: default 'test'
    test_subset = 'test'

    # Below are parameters for text classification/structural prediction
    # Required: prompt template:
    # e.g  `sentence` is the column name for the raw text "It was" and "." are templates, <mask> is the masked poition
    # Then the template is "<text> It was <mask>." <e> is for the entity tokens
    # ex: Obama was the president of USA. Obama is [MASK] [MASK] entity. (ans: a person)
    templates = [
        'sentence|<e>|is |<mask>|entity.',
    ]
    # Required: The label for this task
    labels = ['O', 'corporation', 'creative-work', 'group', 'location', 'person', 'product']
    # Required: Manually-designed label for each class, the order of `labels` and label_mapping` should match
    tags = {'O': 0, 'B-corporation': 1, 'I-corporation': 2,
            'B-creative-work': 3, 'I-creative-work': 4, 'B-group': 5, ' I-group': 6,
            'B-location': 7, 'I-location': 8, 'B-person': 9, 'I-person': 10, 'B-product': 11, 'I-product': 12}
    label_mappings = [
        [
            ['not an'],
            ['a corporation'],
            ['a creative'],
            ['a group'],
            ['a location'],
            ['a person'],
            ['a product'],
        ]
    ]
    # Optional: specify the name of the label column
    label_name = 'label'
    sentence_label = 'sentence'
    remove_punc = False
    data_preprocessing = processing_conll2003
    max_span_width = 6



@register_task('ncbi_disease')
class ConllConfig(BaseConfig):
    # Required: The unique task identifier for this task
    task = 'ncbi_disease'
    # Required: the task type, each task type corresponds to a data processor
    task_type = 'structure_prediction'
    # Required: Either input a file name that can be tracked in the environment(like  '${PATH_TO_FILE}/${FILE_NAME}')
    # or a str or list, which is a dataset name for huggingface's `datasets`
    dataset_name = 'ncbi_disease'  # datasets.load_dataset('glue', 'sst2')
    # dataset_name = 'lama.json'  # datasets.load_dataset('json', 'lama.json')
    # dataset_name = 'lama '    # datasets.load_dataset('lama')
    # Required: The metrics used for this task, using metrics in huggingface.Metrics or defined metrics in metrics.py
    metrics = 'f1-ner'
    # metrics_kwargs = {'average': 'micro'}
    # Optional: The data split used for evaluation: default 'test'
    test_subset = 'test'

    # Below are parameters for text classification/structural prediction
    # Required: prompt template:
    # e.g  `sentence` is the column name for the raw text "It was" and "." are templates, <mask> is the masked poition
    # Then the template is "<text> It was <mask>." <e> is for the entity tokens
    # ex: Obama was the president of USA. Obama is [MASK] [MASK] entity. (ans: a person)
    templates = [
        'sentence|<e>|is |<mask>|entity.',
    ]
    # Required: The label for this task
    labels = ['O', 'disease']
    # Required: Manually-designed label for each class, the order of `labels` and label_mapping` should match
    tags = {'O': 0, 'B-disease': 1, 'I-disease': 2}
    label_mappings = [
        [
            ['not an'],
            ['a disease'],
        ]
    ]
    # Optional: specify the name of the label column
    label_name = 'label'
    sentence_label = 'sentence'
    remove_punc = False
    data_preprocessing = processing_conll2003
    max_span_width = 6
    eval_batch_size = 8

@register_task('conll2000')
class ConllConfig(BaseConfig):
    # Required: The unique task identifier for this task
    task = 'conll2000'
    # Required: the task type, each task type corresponds to a data processor
    task_type = 'structure_prediction'
    # Required: Either input a file name that can be tracked in the environment(like  '${PATH_TO_FILE}/${FILE_NAME}')
    # or a str or list, which is a dataset name for huggingface's `datasets`
    dataset_name = 'conll2000'  # datasets.load_dataset('glue', 'sst2')
    # dataset_name = 'lama.json'  # datasets.load_dataset('json', 'lama.json')
    # dataset_name = 'lama '    # datasets.load_dataset('lama')
    # Required: The metrics used for this task, using metrics in huggingface.Metrics or defined metrics in metrics.py
    metrics = 'f1-ner'
    # metrics_kwargs = {'average': 'micro'}
    # Optional: The data split used for evaluation: default 'test'
    test_subset = 'test'

    # Below are parameters for text classification/structural prediction
    # Required: prompt template:
    # e.g  `sentence` is the column name for the raw text "It was" and "." are templates, <mask> is the masked poition
    # Then the template is "<text> It was <mask>." <e> is for the entity tokens
    # ex: Obama was the president of USA. Obama is [MASK] [MASK] entity. (ans: a person)
    templates = [
        'sentence|<e>|is |<mask>|phrase.',
    ]
    # Required: The label for this task
    labels = ['O', 'ADJP', 'ADVP', 'CONJP', 'INTJ', 'LST', 'NP', 'PP', 'PRT', 'SBAR', 'UCP', 'VP']
    # Required: Manually-designed label for each class, the order of `labels` and label_mapping` should match
    tags = {'O': 0, 'B-ADJP': 1, 'I-ADJP': 2, 'B-ADVP': 3, 'I-ADVP': 4, 'B-CONJP': 5, 'I-CONJP': 6,
            'B-INTJ': 7, 'I-INTJ': 8, 'B-LST': 9, 'I-LST': 10, 'B-NP': 11, 'I-NP': 12, 'B-PP': 13, 'I-PP': 14,
            'B-PRT': 15, 'I-PRT': 16, 'B-SBAR': 17, 'I-SBAR': 18, 'B-UCP': 19, 'I-UCP': 20, 'B-VP': 21, 'I-VP': 22}
    label_mappings = [
        [
            ['not a'],
            ['an adjective'],
            ['an adverb'],
            ['a conjunction'],
            ['an interjection'],
            ['a marker'],  # ['a list marker'],
            ['a noun'],
            ['a prepositional'],
            ['a particle'],
            ['a clause'],  # ['a subordinate clause'],
            ['a coordinate'],  # ['a unlike coordinated'],
            ['a verb'],
        ]
    ]
    # Optional: specify the name of the label column
    label_name = 'label'
    sentence_label = 'sentence'
    remove_punc = False
    data_preprocessing = processing_conll2000
    # spanlen_num_dic:  {1: 13233, 2: 6266, 3: 2864, 4: 944, 5: 326, 6: 139, 7: 39, 8: 16, 9: 8, 10: 5, 11: 1}
    max_span_width = 6
    '''
    evaluate on 100 samples of conll2000...
    Using template: sentence|<e>|is |<mask>|phrase.
    plm : roberta-large
    f1 : 0.005253940455341506
    precision : 0.058823529411764705
    recall : 0.002749770852428964
    '''

@register_task('conll2003_chunk')
class ConllConfig(BaseConfig):
    # Required: The unique task identifier for this task
    task = 'conll2003_chunk'
    # Required: the task type, each task type corresponds to a data processor
    task_type = 'structure_prediction'
    # Required: Either input a file name that can be tracked in the environment(like  '${PATH_TO_FILE}/${FILE_NAME}')
    # or a str or list, which is a dataset name for huggingface's `datasets`
    dataset_name = 'conll2003'  # datasets.load_dataset('glue', 'sst2')
    # dataset_name = 'lama.json'  # datasets.load_dataset('json', 'lama.json')
    # dataset_name = 'lama '    # datasets.load_dataset('lama')
    # Required: The metrics used for this task, using metrics in huggingface.Metrics or defined metrics in metrics.py
    metrics = 'f1-ner'
    # metrics_kwargs = {'average': 'micro'}
    # Optional: The data split used for evaluation: default 'test'
    test_subset = 'test'

    # Below are parameters for text classification/structural prediction
    # Required: prompt template:
    # e.g  `sentence` is the column name for the raw text "It was" and "." are templates, <mask> is the masked poition
    # Then the template is "<text> It was <mask>." <e> is for the entity tokens
    # ex: Obama was the president of USA. Obama is [MASK] [MASK] entity. (ans: a person)
    templates = [
        'sentence|<e>|is |<mask>|phrase.',
    ]
    # Required: The label for this task
    labels = ['O', 'ADJP', 'ADVP', 'CONJP', 'INTJ', 'LST', 'NP', 'PP', 'PRT', 'SBAR', 'UCP', 'VP']
    # Required: Manually-designed label for each class, the order of `labels` and label_mapping` should match
    tags = {'O': 0, 'B-ADJP': 1, 'I-ADJP': 2, 'B-ADVP': 3, 'I-ADVP': 4, 'B-CONJP': 5, 'I-CONJP': 6,
            'B-INTJ': 7, 'I-INTJ': 8, 'B-LST': 9, 'I-LST': 10, 'B-NP': 11, 'I-NP': 12, 'B-PP': 13, 'I-PP': 14,
            'B-PRT': 15, 'I-PRT': 16, 'B-SBAR': 17, 'I-SBAR': 18, 'B-UCP': 19, 'I-UCP': 20, 'B-VP': 21, 'I-VP': 22}
    label_mappings = [
        [
            ['not a'],
            ['an adjective'],
            ['an adverb'],
            ['a conjunction'],
            ['an interjection'],
            ['a marker'],  # ['a list marker'],
            ['a noun'],
            ['a prepositional'],
            ['a particle'],
            ['a clause'],  # ['a subordinate clause'],
            ['a coordinate'],  # ['a unlike coordinated'],
            ['a verb'],
        ]
    ]
    # Optional: specify the name of the label column
    label_name = 'label'
    sentence_label = 'sentence'
    remove_punc = False
    data_preprocessing = processing_conll2003_chunk
    max_span_width = 6
    '''
    evaluate on 100 samples of conll2003-chunk...
    Using template: sentence|<e>|is |<mask>|phrase.
    plm : roberta-large
    f1 : 0.04772234273318872
    precision : 0.08906882591093117
    recall : 0.03259259259259259
    '''

@register_task('conll2003_pos')
class ConllConfig(BaseConfig):
    # Required: The unique task identifier for this task
    task = 'conll2003_pos'
    # Required: the task type, each task type corresponds to a data processor
    task_type = 'structure_prediction'
    # Required: Either input a file name that can be tracked in the environment(like  '${PATH_TO_FILE}/${FILE_NAME}')
    # or a str or list, which is a dataset name for huggingface's `datasets`
    dataset_name = 'conll2003'  # datasets.load_dataset('glue', 'sst2')
    # dataset_name = 'lama.json'  # datasets.load_dataset('json', 'lama.json')
    # dataset_name = 'lama '    # datasets.load_dataset('lama')
    # Required: The metrics used for this task, using metrics in huggingface.Metrics or defined metrics in metrics.py
    metrics = 'f1-ner' # accuracy
    # metrics_kwargs = {'average': 'micro'}
    # Optional: The data split used for evaluation: default 'test'
    test_subset = 'test'

    # Below are parameters for text classification/structural prediction
    # Required: prompt template:
    # e.g  `sentence` is the column name for the raw text "It was" and "." are templates, <mask> is the masked poition
    # Then the template is "<text> It was <mask>." <e> is for the entity tokens
    # ex: Obama was the president of USA. Obama is [MASK] [MASK] entity. (ans: a person)
    templates = [
        'sentence|<e>|is |<mask>|word.',
    ]
    # Required: The label for this task
    labels = ['"', "''", '#', '$', '(', ')', ',', '.', ':', '``', 'CC', 'CD', 'DT',
         'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS',
         'NNS', 'NN|SYM', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP',
         'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT','WP', 'WP$', 'WRB']
    # Required: Manually-designed label for each class, the order of `labels` and label_mapping` should match
    tags = {'"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9, 'CC': 10, 'CD': 11, 'DT': 12,
         'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21, 'NNP': 22, 'NNPS': 23,
         'NNS': 24, 'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33,
         'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43,
         'WP': 44, 'WP$': 45, 'WRB': 46}
    label_mappings = [
        [
            ['a double quote'],
            ['a single quote'],
            ['a number sign'],
            ['a dollar sign'],
            ['a left parenthesis'],
            ['a right parenthesis'],
            ['a apostrophe sign'],
             ['a comma sign'],
             ['a colon sign'],
             ['a tandem colon'],
             ['a coordinating conjunction'],
             ['a cardinal number'],
             ['a determiner'],
             ['an existential there'],
             ['a foreign'],
             ['a preposition conjunction'],
             ['an adjective'],
             ['a comparative adjective'],
             ['a superlative comparative'],
             ['a list marker'],
             ['a modal'],
             ['a singular noun'],
             ['a singular proper noun'],
             ['a plural proper noun'],
             ['a plural noun'],
             ['a singular noun'],
             ['a predeterminer'],
             ['a possessive ending'],
             ['a personal pronoun'],
             ['a possessive pronoun'],
             ['an adverb'],
             ['a comparative adverb'],
             ['a superlative adverb'],
             ['a particle'],
             ['a symbol'],
             ['an infinite marker'],
             ['an interjection'],
             ['a base form verb'],
             ['a past tense verb'],
             ['a gerund verb'],
             ['a past participle verb'],
             ['a non-3rd person verb'],
             ['a 3rd person verb'],
             ['a wh-determiner'],
             ['a wh-pronoun'],
             ['a possessive wh-pronoun'],
             ['a wh-adverb'],
             ]
    ]
    # Optional: specify the name of the label column
    label_name = 'label'
    sentence_label = 'sentence'
    remove_punc = False
    data_preprocessing = processing_conll2003_pos
    max_span_width = 1



