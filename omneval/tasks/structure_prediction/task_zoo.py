from omneval.tasks import BaseConfig
from omneval.registry import register_task
from omneval.utils import make_sentence
import pandas as pd
from datasets import Dataset


def processing_conll2003(config, data):
    examples = []
    for item in data:
        sentence = make_sentence(item['tokens'])
        for token, tag in zip(item['tokens'], item['ner_tags']):
            examples.append({'sentence': sentence, 'token': token.lower(), 'label': tag})
    return Dataset.from_pandas(pd.DataFrame(examples))


@register_task('conll2003')
class SemEvalConfig(BaseConfig):
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
    metrics = 'f1'
    metrics_kwargs = {'average': 'micro'}
    # Optional: The data split used for evaluation: default 'test'
    test_subset = 'test'

    # Below are parameters for text classification/structural prediction
    # Required: prompt template:
    # e.g  `sentence` is the column name for the raw text "It was" and "." are templates, <mask> is the masked poition
    # Then the template is "<text> It was <mask>." <e> is for the entity tokens
    # ex: Obama was the president of USA. Obama is the [mask] word of [mask] entity. (ans: beginning, person)
    templates = [
        'sentence|<e>|is the|<mask>|word of |<mask>|entity.',
    ]
    # Required: The label for this task
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # Required: Manually-designed label for each class, the order of `labels` and label_mapping` should match
    label_mappings = [
        ['outside', 'any'],
        ['beginning', 'person'],
        ['inside', 'person'],
        ['beginning', 'organization'],
        ['inside', 'organization'],
        ['beginning', 'location'],
        ['inside', 'location'],
        ['beginning', 'other'],
        ['inside', 'other'],
    ]
    # Optional: specify the name of the label column
    label_name = 'label'
    sentence_label = 'sentence'
    remove_punc = False
    data_preprocessing = processing_conll2003
