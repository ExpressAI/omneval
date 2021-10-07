import torch
import warnings
from .. import register_processor
from .. import BaseProcessor
import logging
from transformers import GPT2Tokenizer, GPT2TokenizerFast

warnings.filterwarnings('ignore')

def difference(list1, list2):
    return [item for item in list1 if item not in list2]


@register_processor('knowledge_probing')
class ProcessorForKnowledgeProbing(BaseProcessor):

    def __init__(self, arch, config):
        super(ProcessorForKnowledgeProbing, self).__init__(arch, config)

    @property
    def prompt_count(self):
        return 1   # For knowledge probing datasets, the templates are integrated in the dataset

    def generate_dataset(self, prompt_order=0):
        padding_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        query = getattr(self.config, 'query_name', 'template')
        subject = getattr(self.config, 'subject_name', 'sub_label')
        label = getattr(self.config, 'label_name', 'obj_label')
        remove_columns = difference(self.raw_data.features.keys(), [label])
        def prompting(example, tokenizer, max_seq_length, padding_id=0):
            mask_token = tokenizer.mask_token if tokenizer._mask_token is not None else tokenizer.unk_token
            mask_token_id = tokenizer.mask_token_id if tokenizer._mask_token is not None else tokenizer.unk_token_id
            text = example[query].replace('[X]', example[subject]).replace('[Y]', mask_token)

            res = tokenizer(text.strip())
            text_len = len(res['input_ids'])
            res['mask_pos'] = [0] * max_seq_length
            mask_start_pos = res['input_ids'].index(mask_token_id)
            # TODO: is this task only for single token classification?
            res['mask_pos'][mask_start_pos] = 1
            res['input_ids'] += (max_seq_length - text_len) * [padding_id]
            res['attention_mask'] += (max_seq_length - text_len) * [0]
            if res.get('token_type_ids'):
                res['token_type_ids'] += (max_seq_length - text_len) * [0]
            return res

        return self.raw_data.map(
            lambda x: prompting(example=x,
                                tokenizer=self.tokenizer,
                                max_seq_length=getattr(self.config, "max_seq_length", 512),
                                padding_id=padding_id),
            remove_columns=remove_columns)

