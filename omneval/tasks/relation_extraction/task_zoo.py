from omneval.tasks import BaseConfig
from omneval.registry import register_task

@register_task('sem_eval_2010_task_8')
class SemEvalConfig(BaseConfig):
    # Required: The unique task identifier for this task
    task = 'sem_eval_2010_task_8'
    # Required: the task type, each task type corresponds to a data processor
    task_type = 'relation_extraction'
    # Required: Either input a file name that can be tracked in the environment(like  '${PATH_TO_FILE}/${FILE_NAME}')
    # or a str or list, which is a dataset name for huggingface's `datasets`
    dataset_name = ['sem_eval_2010_task_8']  # datasets.load_dataset('glue', 'sst2')
    # dataset_name = 'lama.json'  # datasets.load_dataset('json', 'lama.json')
    # dataset_name = 'lama '    # datasets.load_dataset('lama')
    # Required: The metrics used for this task
    metrics = 'f1'
    metrics_kwargs = {'average': 'micro'}
    # Optional: The data split used for evaluation: default 'test'
    test_subset = 'test'
    # Below are parameters for text classification
    # Required: prompt template:
    # e.g  `sentence` is the column name for the raw text "It was" and "." are templates, <mask> is the masked poition
    # Then the template is "<text> It was <mask>."
    templates = [
        'sentence|The|<mask>|<e1>|was|<mask>|to the|<mask>|<e2>|.',
    ]
    # Required: The label for this task
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    # Required: Manually-designed label for each class, the order of `labels` and label_mapping` should match
    label_mappings = [
        ['member', 'related', 'collection'],
        ['collection', 'related', 'member'],
        ['entity', 'related', 'origin'],
        ['origin', 'related', 'entity'],
        ['cause', 'related', 'effect'],
        ['effect', 'related', 'cause'],
        ['component', 'related', 'whole'],
        ['whole', 'related', 'component'],
        ['product', 'related', 'producer'],
        ['producer', 'related', 'product'],
        ['instrument', 'related', 'agency'],
        ['agency', 'related', 'instrument'],
        ['entity', 'related', 'destination'],
        ['destination', 'related', 'entity'],
        ['content', 'related', 'container'],
        ['container', 'related', 'content'],
        ['message', 'related', 'topic'],
        ['topic', 'related', 'message'],
        ['mention', 'irrelevant', 'mention'],
    ]
    # Optional: specify the name of the label column
    label_name = 'relation'
    sentence_label = 'sentence'
    remove_punc = False


@register_task('tacred')
class SemEvalConfig(BaseConfig):
    # Required: The unique task identifier for this task
    task = 'tacred.csv'
    # Required: the task type, each task type corresponds to a data processor
    task_type = 'relation_extraction'
    # Required: Either input a file name that can be tracked in the environment(like  '${PATH_TO_FILE}/${FILE_NAME}')
    # or a str or list, which is a dataset name for huggingface's `datasets`
    dataset_name = 'tacred.csv'  # datasets.load_dataset('glue', 'sst2')
    # dataset_name = 'lama.json'  # datasets.load_dataset('json', 'lama.json')
    # dataset_name = 'lama '    # datasets.load_dataset('lama')
    # Required: The metrics used for this task
    metrics = 'f1'
    metrics_kwargs = {'average': 'micro'}
    # Optional: The data split used for evaluation: default 'test'
    test_subset = 'test'
    # Below are parameters for text classification
    # Required: prompt template:
    # e.g  `sentence` is the column name for the raw text "It was" and "." are templates, <mask> is the masked poition
    # Then the template is "<text> It was <mask>."
    templates = [
        'text|The|<mask>|<e1>|<mask>|the|<mask>|<e2>|.',
    ]
    # Required: The label for this task
    labels =['per:charges',
             'per:date_of_death',
             'per:country_of_death',
             'per:cause_of_death',
             'org:founded_by',
             'org:founded',
             'per:city_of_death',
             'per:stateorprovince_of_death',
             'per:date_of_birth',
             'per:stateorprovince_of_birth',
             'per:country_of_birth',
             'per:city_of_birth',
             'org:shareholders',
             'per:other_family',
             'per:title',
             'org:dissolved',
             'org:stateorprovince_of_headquarters',
             'org:country_of_headquarters',
             'org:city_of_headquarters',
             'per:countries_of_residence',
             'per:stateorprovinces_of_residence',
             'per:cities_of_residence',
             'org:member_of',
             'per:religion',
             'org:political/religious_affiliation',
             'org:top_members/employees',
             'org:number_of_employees/members',
             'per:schools_attended',
             'per:employee_of',
             'per:siblings',
             'per:spouse',
             'per:parents',
             'per:children',
             'per:alternate_names',
             'org:alternate_names',
             'org:members',
             'org:parents',
             'org:subsidiaries',
             'per:origin',
             'org:website',
             'per:age',
             'no_relation']
    # Required: Manually-designed label for each class, the order of `labels` and label_mapping` should match
    label_mappings = [['person', 'was charged with', 'event'],
                     ['person', 'was died on', 'date'],
                     ['person', 'was died in', 'country'],
                     ['person', 'was died of', 'event'],
                     ['organization', 'was founded by', 'person'],
                     ['organization', 'was founded in', 'date'],
                     ['person', 'was died in', 'city'],
                     ['person', 'was died in', 'state'],
                     ['person', 'was born in', 'date'],
                     ['person', 'was born in', 'state'],
                     ['person', 'was born in', 'country'],
                     ['person', 'was born in', 'city'],
                     ['organization', 'was invested by', 'person'],
                     ['person', "'s relative is", 'person'],
                     ['person', "'s title is", 'title'],
                     ['organization', 'was dissolved in', 'date'],
                     ['organization', 'was located in', 'state'],
                     ['organization', 'was located in', 'country'],
                     ['organization', 'was located in', 'city'],
                     ['person', 'was living in', 'country'],
                     ['person', 'was living in', 'state'],
                     ['person', 'was living in', 'city'],
                     ['organization', 'was member of', 'organization'],
                     ['person', 'was member of', 'religion'],
                     ['organization', 'was member of', 'religion'],
                     ['organization', "'s employer was", 'person'],
                     ['organization', "'s employer has", 'number'],
                     ['person', "'s school was", 'organization'],
                     ['person', "'s employee was", 'organization'],
                     ['person', "'s sibling was", 'person'],
                     ['person', "'s spouse was", 'person'],
                     ['person', "'s parent was", 'person'],
                     ['person', "'s child was", 'person'],
                     ['person', "'s alias was", 'person'],
                     ['organization', "'s alias was", 'organization'],
                     ['organization', "'s member was", 'organization'],
                     ['organization', "'s parent was", 'organization'],
                     ['organization', "'s subsidiary was", 'organization'],
                     ['person', "'s nationality was", 'country'],
                     ['organization', "'s website was", 'link'],
                     ['person', "'s age was", 'number'],
                     ['entity', 'is irrelevant to', 'entity']]
    # Optional: specify the name of the label column
    label_name = 'relation'
    sentence_label = 'text'
    remove_punc = False