import torch
import warnings
from .. import register_processor
from .. import BaseProcessor
import logging
import pdb

warnings.filterwarnings('ignore')


@register_processor('classification')
class ProcessorForClassification(BaseProcessor):

    def __init__(self, arch, config):
        assert hasattr(config, 'templates'), "prompt templates should be specified."
        assert hasattr(config, 'label_mappings'), "label mappings should be specified."
        super(ProcessorForClassification, self).__init__(arch, config)

    @property
    def prompt_count(self):
        return len(self.config.templates) * len(self.config.label_mappings)

    def generate_dataset(self, prompt_order=0):
        padding_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        order_t, order_l = prompt_order // len(self.config.label_mappings), prompt_order % len(
            self.config.label_mappings)
        prompt_schema, verbalizers = self.config.templates[order_t], self.config.label_mappings[order_l]
        #TODO: Add torch logger
        info = "Using %s, and label mappings %s"%(prompt_schema, str(verbalizers))
        logging.info(info)
        mask_length = len(self.tokenizer.tokenize(verbalizers[0]))
        for v in verbalizers:
            if mask_length != len(self.tokenizer.tokenize(v)):
                raise ValueError("Currently the framework only supports label mappings of the same token sizes")

        def prompting(example, tokenizer, prompt_schema, mask_length, max_seq_length, padding_id=0):
            text = ''
            mask_token = tokenizer.mask_token if tokenizer._mask_token is not None else tokenizer.unk_token
            mask_token_id = tokenizer.mask_token_id if tokenizer._mask_token is not None else tokenizer.unk_token_id
            remove_punc = getattr(self.config, "remove_punc", False)

            for item in prompt_schema.split('|'):
                item = item.strip()
                # for the placeholders of verbalizers
                if item == '<mask>':
                    # TODO: Should find better adaptation ways for different tokenizers(GPT2)
                    if mask_token != '<|endoftext|>':
                        if text and text[-1] != ' ':
                            text += ' '
                        text += (mask_token + ' ') * mask_length
                    else:
                        text += mask_token * mask_length
                    # for raw inputs
                elif example.get(item) and isinstance(example[item], str):
                    if remove_punc and example[item][-1] in '.,?!':
                        text += example[item][: -1]
                    else:
                        text += example[item]
                # for prompting templates
                else:
                    if text and text[-1] != ' ':
                        text += ' '
                    text += item
                    if text and text[-1] != ' ':
                        text += ' '
            res = tokenizer(text.strip())
            text_len = len(res['input_ids'])
            res['mask_pos'] = [0] * max_seq_length
            mask_start_pos = res['input_ids'].index(mask_token_id)
            for i in range(mask_start_pos, mask_start_pos + mask_length):
                res['mask_pos'][i] = 1
            res.update({'label': int(round(example['label']))})
            res['input_ids'] += (max_seq_length - text_len) * [padding_id]
            res['attention_mask'] += (max_seq_length - text_len) * [0]
            if res.get('token_type_ids'):
                res['token_type_ids'] += (max_seq_length - text_len) * [0]
            return res

        return self.raw_data.map(
            lambda x: prompting(example=x,
                                tokenizer=self.tokenizer,
                                prompt_schema=prompt_schema,
                                mask_length=mask_length,
                                max_seq_length=getattr(self.config, "max_seq_length", 512),
                                padding_id=padding_id),
            remove_columns=getattr(self.config, "remove_columns", None))

    def convert_verbalizers_to_ids(self, idx):
        return [list(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x))) for x in
                self.config.label_mappings[idx]]

    def generate_aux_inputs(self, prompt_order=0):
        order_t, order_l = prompt_order // len(self.config.label_mappings), prompt_order % len(
            self.config.label_mappings)
        candidate_idx = torch.LongTensor(self.convert_verbalizers_to_ids(order_l))
        candidate_labels = self.config.labels
        return {'candidate_idx': candidate_idx, 'candidate_labels': candidate_labels}
