from omneval.tasks import BaseConfig
from omneval.registry import register_task
import pdb

def processing_winogrande(config, data):
    def merge_options(example):
        example['options'] = [example['option1'], example['option2']]
        return example
    return data.map(merge_options, remove_columns=['option1', 'option2'])

@register_task('winogrande_xs')
class WinograndeConfig(BaseConfig):
    task = 'winogrande_xs'
    task_type = 'multiple_choice'
    dataset_name = ['winogrande', 'winogrande_xs']
    metrics = 'accuracy'
    test_subset = 'validation'
    options_name = 'options'
    label_name = 'answer'
    labels = ['1', '2']
    templates = ['sentence']
    mask_replace_column = 'sentence'
    mask_replace_token = '_'
    data_preprocessing = processing_winogrande


def processing_olympics(config, data):
    def merge_options(example):
        example['options'] = [0] * len(config.labels)
        for item in example['question']['choices']:
            example['options'][config.answer_choices.index(item['label'])] = item['text']
        example['sentence'] = example['question']['stem']
        example[config.label_name] = config.answer_choices.index(example[config.label_name])
        return example

    return data.map(merge_options, remove_columns=['question', 'id'])


@register_task('olmpics_always_never')
class Olmpic1Config(BaseConfig):
    task = 'olmpics_always_never'
    task_type = 'multiple_choice'
    dataset_name = 'olmpics/coffee_cats_quantifiers_dev.json'
    metrics = 'accuracy'
    options_name = 'options'
    label_name = 'answerKey'
    labels = [0,1,2,3,4]
    answer_choices = ['A', 'B', 'C', 'D', 'E']
    templates = ['sentence']
    mask_replace_column = 'sentence'
    mask_replace_token = '[MASK]'
    data_preprocessing = processing_olympics

# TODO: huggingface.metrics.accuracy does not support non integer labels
@register_task('olmpics_age_comparison')
class Olmpic1Config(BaseConfig):
    task = 'olmpics_age_comparison'
    task_type = 'multiple_choice'
    dataset_name = 'olmpics/number_comparison_age_compare_masked_dev.json'
    metrics = 'accuracy'
    options_name = 'options'
    label_name = 'answerKey'
    labels = [0,1]
    answer_choices = ['A', 'B']
    templates = ['sentence']
    mask_replace_column = 'sentence'
    mask_replace_token = '[MASK]'
    data_preprocessing = processing_olympics


@register_task('olmpics_objects_comparison')
class Olmpic1Config(BaseConfig):
    task = 'olmpics_objects_comparison'
    task_type = 'multiple_choice'
    dataset_name = 'olmpics/size_comparison_dev.json'
    metrics = 'accuracy'
    options_name = 'options'
    label_name = 'answerKey'
    labels = [0,1]
    answer_choices = ['A', 'B']
    templates = ['sentence']
    mask_replace_column = 'sentence'
    mask_replace_token = '[MASK]'
    data_preprocessing = processing_olympics


@register_task('olmpics_antonym_negation')
class Olmpic1Config(BaseConfig):
    task = 'olmpics_antonym_negation'
    task_type = 'multiple_choice'
    dataset_name = 'olmpics/antonym_synonym_negation_dev.json'
    metrics = 'accuracy'
    options_name = 'options'
    label_name = 'answerKey'
    labels = [0,1]
    answer_choices = ['A', 'B']
    templates = ['sentence']
    mask_replace_column = 'sentence'
    mask_replace_token = '[MASK]'
    data_preprocessing = processing_olympics


@register_task('olmpics_property_conjunction')
class Olmpic1Config(BaseConfig):
    task = 'olmpics_property_conjunction'
    task_type = 'multiple_choice'
    dataset_name = 'olmpics/conjunction_filt4_dev.json'
    metrics = 'accuracy'
    options_name = 'options'
    label_name = 'answerKey'
    labels = [0, 1, 2]
    answer_choices = ['A', 'B', 'C']
    templates = ['sentence']
    mask_replace_column = 'sentence'
    mask_replace_token = '[MASK]'
    data_preprocessing = processing_olympics


@register_task('olmpics_taxonomy_conjunction')
class Olmpic1Config(BaseConfig):
    task = 'olmpics_taxonomy_conjunction'
    task_type = 'multiple_choice'
    dataset_name = 'olmpics/hypernym_conjunction_dev.json'
    metrics = 'accuracy'
    options_name = 'options'
    label_name = 'answerKey'
    labels = [0, 1, 2]
    answer_choices = ['A', 'B', 'C']
    templates = ['sentence']
    mask_replace_column = 'sentence'
    mask_replace_token = '[MASK]'
    data_preprocessing = processing_olympics


@register_task('olmpics_encyclopedic_composition')
class Olmpic1Config(BaseConfig):
    task = 'olmpics_encyclopedic_composition'
    task_type = 'multiple_choice'
    dataset_name = 'olmpics/composition_v2_dev.json'
    metrics = 'accuracy'
    options_name = 'options'
    label_name = 'answerKey'
    labels = [0, 1, 2]
    answer_choices = ['A', 'B', 'C']
    templates = ['sentence']
    mask_replace_column = 'sentence'
    mask_replace_token = '[MASK]'
    data_preprocessing = processing_olympics


@register_task('olmpics_multihop_composition')
class Olmpic1Config(BaseConfig):
    task = 'olmpics_multihop_composition'
    task_type = 'multiple_choice'
    dataset_name = 'olmpics/compositional_comparison_dev.json'
    metrics = 'accuracy'
    options_name = 'options'
    label_name = 'answerKey'
    labels = [0, 1, 2]
    answer_choices = ['A', 'B', 'C']
    templates = ['sentence']
    mask_replace_column = 'sentence'
    mask_replace_token = '[MASK]'
    data_preprocessing = processing_olympics