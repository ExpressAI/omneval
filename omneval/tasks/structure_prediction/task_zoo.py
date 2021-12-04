from omneval.tasks import BaseConfig
from omneval.registry import register_task
from omneval.utils import make_sentence, get_entity_span_ids
import pandas as pd
from datasets import Dataset
import pdb
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans



def processing_conll2003(config, data):
    examples = []
    for idx, item in enumerate(data):
        # TODO: Do we need to normalize each vocabulary of the text?
        # Only to test
        if idx > 500:
            break
        sentence = ' '.join(item['tokens'])
        span_ids = set(enumerate_spans(item['tokens'], max_span_width=config.max_span_width))
        entity_span_ids = get_entity_span_ids(item['ner_tags'], config.tags)
        for label, start_id, end_id in entity_span_ids:
            span = item['tokens'][start_id: end_id+1]
            examples.append({'sentence_id': idx, 'span': span, 'label': label, 'sentence': sentence,
                             'span_idx': (start_id, end_id+1)})
            span_ids.discard((start_id, end_id))
        for start_id, end_id in span_ids:
            span = item['tokens'][start_id: end_id+1]
            examples.append({'sentence_id': idx, 'span': span, 'label': 'O', 'sentence': sentence,
                             'span_idx': (start_id, end_id+1)})
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
            ['a location'],
            ['a person'],
            ['an organization'],
            ['an other'],
        ]
    ]
    # Optional: specify the name of the label column
    label_name = 'label'
    sentence_label = 'sentence'
    remove_punc = False
    data_preprocessing = processing_conll2003
    max_span_width = 3
