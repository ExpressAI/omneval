import torch
import warnings
from omneval.tasks import BaseProcessor
from omneval.utils import pad_input_ids, truncate_text, normalize_raw_text_to_inputs, \
    append_templates_to_inputs, append_mask_token_to_inputs, difference, check_if_bpe_tokenizer, replace_tokens_to_mask
from omneval.registry import register_processor
import pdb
import re

warnings.filterwarnings('ignore')

@register_processor('multiple_choice')
class ProcessorForMultipleChoiceClassification(BaseProcessor):

    def __init__(self, config):
        assert hasattr(config, 'templates'), "prompt templates should be specified."
        super(ProcessorForMultipleChoiceClassification, self).__init__(config)
        self.remove_punc = getattr(self.config, "remove_punc", False)
        self.option_name = getattr(self.config, 'option_name', 'options')
        self.mask_replace_token = getattr(self.config, "mask_replace_token", False)
        self.mask_replace_column = getattr(self.config, "mask_replace_column", False)

    @property
    def prompt_count(self):
        return len(self.config.templates)

    def prompt_schema(self, pid):
        return self.config.templates[pid]

    def generate_dataset(self, pid=0):
        """Prompting each instance and build dataset directly for the Evaluator"""
        prompt_schema = self.config.templates[pid]
        remove_columns = difference(self.raw_data.features.keys(), [self.label_name])
        calibrate_word = self.generate_calibrate_example(pid)
        prompt_length = sum(calibrate_word['attention_mask'])
        assert (self.mask_replace_token is not None and self.mask_replace_column is not None \
                and self.mask_replace_column in prompt_schema.split('|')) \
            ^ ('<mask>' in prompt_schema.split('|')), "should define either mask_replace_token and <mask> in prompt_schema"
        return self.raw_data.map(
            lambda x: self.prompting(example=x,
                                     prompt_schema=prompt_schema,
                                     max_length=self.max_seq_length - prompt_length),
            remove_columns=remove_columns) # Since the masked length differs for each example, we give

    def prompting(self, example, prompt_schema, max_length=512):
        text = ''
        text_length_cnt = 0
        if self.option_name in example:
            labels_ids, labels_masks, mask_length = self.convert_verbalizers_to_ids(example[self.option_name])
        else:
            labels_ids, labels_masks, mask_length = 0, 0, 3

        if self.mask_replace_column:
            max_length -= mask_length
        for item in prompt_schema.split('|'):
            item = item.strip()
            # for the placeholders of label_tokens
            if item == '<mask>':
                text = append_mask_token_to_inputs(text, self.mask_token, mask_length)
            elif item in example and isinstance(example[item], str):
                appended_text = normalize_raw_text_to_inputs(example[item], self.remove_punc)
                appended_text, appended_length = truncate_text(appended_text, self.tokenizer, max_length-text_length_cnt)
                if item == self.mask_replace_column:
                    mask_tokens = self.mask_token * mask_length
                    appended_text = replace_tokens_to_mask(appended_text, self.mask_replace_token, mask_tokens)
                    appended_length += mask_length
                text_length_cnt += appended_length
                text += appended_text
            # for prompting templates
            else:
                text = append_templates_to_inputs(text, item)
        res = self.tokenizer(text.strip())
        text_len = len(res['input_ids'])
        res = pad_input_ids(res, self.max_seq_length, self.padding_id)
        res['mask_pos'] = [0] * self.max_seq_length
        # If no mask in the template, append mask in the end
        try:
            mask_start_pos = res['input_ids'].index(self.mask_token_id)
        except:
            if not getattr(self.config, 'qa_prompting', False):
                mask_start_pos = len(res['input_ids'])
                res['input_ids'] += mask_length * [self.mask_token_id, ]
                res['attention_mask'] += mask_length * [1]
                text_len += mask_length
            else:
                res['decoder_input_ids'] = self.tokenizer.encode(((self.mask_token + ' ') * mask_length).strip())
                mask_start_pos = res['decoder_input_ids'].index(self.mask_token_id)
        try:
            for i in range(mask_start_pos, mask_start_pos + mask_length):
                res['mask_pos'][i] = 1
        except:
            pdb.set_trace()
        ## TODO: this part may be wrong for other tasks
        res.update({self.label_name: example[self.label_name] if example[self.label_name] else 0})
        # Store the labels and answers to each example
        res['labels_ids'] = labels_ids
        res['labels_masks'] = labels_masks
        res['mask_length'] = mask_length
        return res

    def convert_verbalizers_to_ids(self, options_list):
        """

        :return: [[372], [205], [1099], [6587]] like this, 2 levels of lists
        """
        # create label token matrices
        mask_length = 0
        labels_ids = []
        if check_if_bpe_tokenizer(self.tokenizer):
            options_list = [' '+l.lower() for l in options_list]
        for label in options_list:
            token_ids = self.tokenizer.encode(label,  add_special_tokens=False)
            mask_length = max(mask_length, len(token_ids))
            labels_ids.append(token_ids)
        labels_masks = []
        for label in labels_ids:
            label_mask = [1] * len(label) + [0] * (mask_length-len(label))
            label += [self.padding_id] * (mask_length - len(label))
            labels_masks.append(label_mask)
        return labels_ids, labels_masks, mask_length

    def get_label_tokens(self, pid):
        # For GPT2 tokenizers, we should add a space before each word.
        if isinstance(getattr(self.config, "templates_answers_mapping", None), list) \
            and len(self.config.templates_answers_mapping) > pid:
            label_mappings = self.config.label_mappings[self.config.templates_answers_mapping[pid]]
        else:
            label_mappings = self.config.label_mappings[0]
        if check_if_bpe_tokenizer(self.tokenizer):
            label_tokens = [[' '+l.lower() for l in labels] for labels in label_mappings]
        else:
            label_tokens = label_mappings
        return label_tokens

    def generate_aux_inputs(self, pid=0):
        candidate_labels = self.config.labels
        qa_prompting = getattr(self.config, "qa_prompting", None)
        calibrate_input = self.generate_calibrate_example(pid) if getattr(self.config, "calibrate", False) else None
        return {"qa_prompting": qa_prompting, "calibrate_input": calibrate_input, 'candidate_labels': candidate_labels}


