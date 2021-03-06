import torch
import warnings
from omneval.tasks import BaseProcessor
from omneval.utils import difference, pad_input_ids, normalize_raw_text_to_inputs, truncate_text, \
    append_templates_to_inputs, append_mask_token_to_inputs
from omneval.registry import register_processor
import logging
import pdb
from transformers import GPT2Tokenizer, GPT2TokenizerFast
import re

warnings.filterwarnings('ignore')


@register_processor('relation_extraction')
class ProcessorForRelationExtraction(BaseProcessor):

    def __init__(self, config):
        assert hasattr(config, 'templates'), "prompt templates should be specified."
        assert hasattr(config, 'label_mappings'), "label mappings should be specified."
        super(ProcessorForRelationExtraction, self).__init__(config)
        self.labels_ids, self.labels_masks, self.mask_length = self.convert_verbalizers_to_ids()
        self.remove_punc = getattr(self.config, "remove_punc", False)
        self.sentence_label = getattr(self.config, "sentence_label", 'sentence')

    @property
    def prompt_count(self):
        return len(self.config.templates)

    def prompt_schema(self, pid):
        return self.config.templates[pid]

    def generate_dataset(self, pid=0):
        """Prompting each instance and build dataset directly for the Evaluator"""
        prompt_schema = self.config.templates[pid]
        remove_columns = difference(self.raw_data.features.keys(), self.label_name)
        calibrate_word = self.generate_calibrate_example(pid)
        prompt_length = sum(calibrate_word['attention_mask'])
        # TODO: Calibrate not supported by
        return self.raw_data.map(
            lambda x: self.prompting(example=x,
                                     prompt_schema=prompt_schema,
                                     max_length=self.max_seq_length-prompt_length),
                                     remove_columns=remove_columns,)

    def prompting(self, example, prompt_schema, max_length=512):
        text = ''
        # TODO: Should check if it can be generalized to all RE datasets
        e1 = re.findall(r'<e1>(.+)</e1>', example[self.sentence_label])
        e2 = re.findall(r'<e2>(.+)</e2>', example[self.sentence_label])
        e1 = e1[0] if len(e1) > 0 else ''
        e2 = e2[0] if len(e2) > 0 else ''
        mask_cnt = 0
        text_length_cnt = 0
        for item in prompt_schema.split('|'):
            item = item.strip()
            # for the placeholders of label_tokens
            if item == '<mask>':
                text = append_mask_token_to_inputs(text, self.mask_token, self.mask_length[mask_cnt])
                mask_cnt += 1
            # for raw inputs
            elif example.get(item) and isinstance(example[item], str):
                appended_text = example[item].strip()
                if item == self.sentence_label:
                    appended_text = re.sub(r'(<e1>)|(<e2>)|(</e1>)|(</e2>)', '', appended_text)
                # If example[item] is not the beginning of the sentence, lowercase all words
                appended_text = normalize_raw_text_to_inputs(appended_text, self.config.remove_punc)
                appended_text, appended_length = truncate_text(appended_text, self.tokenizer, max_length-text_length_cnt)
                text_length_cnt += appended_length
                text += appended_text
            # for entities
            elif item in ('<e1>', '<e2>'):
                text = text.strip()
                if text:
                    text += ' '
                text += (e1 if item == '<e1>' else e2)
            # for prompting templates
            else:
                text = append_templates_to_inputs(text, item)
        res = self.tokenizer(text.strip())
        res = pad_input_ids(res, self.max_seq_length, self.padding_id)
        res['mask_pos'] = [0] * self.max_seq_length
        # If no mask in the template, append mask in the end
        mask_length_ttl = sum(self.mask_length)
        for i in range(self.max_seq_length):
            if res['input_ids'][i] == self.mask_token_id:
                res['mask_pos'][i] = 1
                mask_length_ttl -= 1
            if mask_length_ttl == 0:
                break
        res.update({self.label_name: example[self.label_name]})
        return res

    def convert_verbalizers_to_ids(self):
        """

        :return: [[[372], [205]], [[1099], [6587]]] like this
        """
        # create label token matrices
        mask_length = [0, 0, 0]
        labels_ids = []
        for class_labels in self.get_label_tokens():
            class_labels_ids = []
            for idx, label in enumerate(class_labels):
                token_ids = self.tokenizer.encode(label,  add_special_tokens=False)
                mask_length[idx] = max(mask_length[idx], len(token_ids))
                class_labels_ids.append(token_ids)
            labels_ids.append(class_labels_ids)
        labels_masks = []
        labels_ids_tmp = []
        for class_ids in labels_ids:
            class_masks = []
            class_label_tmp = []
            class_label_mask_tmp = []
            for idx, label in enumerate(class_ids):
                label_mask = [1] * len(label) + [0] * (mask_length[idx]-len(label))
                label += [self.padding_id] * (mask_length[idx] - len(label))
                class_label_mask_tmp += label_mask
                class_label_tmp += label
            labels_masks.append(class_label_mask_tmp)
            labels_ids_tmp.append(class_label_tmp)
        return labels_ids_tmp, labels_masks, mask_length

    def get_label_tokens(self):
        # For GPT2 tokenizers, we should add a space before each word.
        if isinstance(self.tokenizer, GPT2Tokenizer) or isinstance(self.tokenizer, GPT2TokenizerFast):
            label_tokens = [[' '+l.lower() for l in labels] for labels in self.config.label_mappings]
        else:
            label_tokens = self.config.label_mappings
        return label_tokens

    def generate_aux_inputs(self, pid=0):
        candidate_labels = self.config.labels
        candidate_idx = torch.LongTensor(self.labels_ids)
        candidate_idx_mask = torch.LongTensor(self.labels_masks)
        mask_length = self.mask_length
        # TODO: Need to check if qa_prompting still need to be used
        qa_prompting = getattr(self.config, "qa_prompting", None)
        calibrate_input = self.generate_calibrate_example(pid) if getattr(self.config, "calibrate", False) else None
        return {'candidate_idx': candidate_idx, 'candidate_idx_mask': candidate_idx_mask,
                'candidate_labels': candidate_labels, "qa_prompting": qa_prompting, "calibrate_input": calibrate_input,
                'mask_length': sum(mask_length)}



