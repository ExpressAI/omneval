import torch
import warnings
from omneval.tasks import BaseProcessor
from omneval.utils import difference, pad_input_ids
from omneval.registry import register_processor
import logging
from transformers import GPT2Tokenizer, GPT2TokenizerFast
import pdb

warnings.filterwarnings('ignore')


@register_processor('knowledge_probing')
class ProcessorForKnowledgeProbing(BaseProcessor):

    def __init__(self, config):
        super(ProcessorForKnowledgeProbing, self).__init__(config)
        self.query = getattr(self.config, 'query_name', 'template')
        self.subject = getattr(self.config, 'subject_name', 'sub_label')
        self.label_name = getattr(self.config, 'label_name', 'obj_label')

    @property
    def prompt_count(self):
        return 1   # For knowledge probing datasets, the templates are integrated in the dataset

    def prompt_schema(self, pid):
        return 'Specified in the dataset'

    def generate_dataset(self, pid=0):
        remove_columns = difference(self.raw_data.features.keys(), self.label_name)
        return self.raw_data.map(
            lambda x: self.prompting(example=x,
                                max_length=self.max_seq_length),
            remove_columns=remove_columns)

    def prompting(self, example, prompt_schema=None, max_length=512):
        text = example[self.query].replace('[X]', example[self.subject]).replace('[Y]', self.mask_token)
        res = self.tokenizer(text.strip())
        res = pad_input_ids(res, self.max_seq_length, self.padding_id)
        res['mask_pos'] = [0] * self.max_seq_length
        mask_start_pos = res['input_ids'].index(self.mask_token_id)
        # TODO: is this task only for single token classification?
        res['mask_pos'][mask_start_pos] = 1
        if isinstance(self.tokenizer, GPT2Tokenizer) or isinstance(self.tokenizer, GPT2TokenizerFast):
            example[self.label_name] = ' '+example[self.label_name]
        res[self.label_name] = self.tokenizer.encode(example[self.label_name], add_special_tokens=False)[0]
        return res