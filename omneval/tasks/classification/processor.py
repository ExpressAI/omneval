import torch
import warnings
from omneval.tasks import BaseProcessor
from omneval.utils import pad_input_ids, truncate_text, normalize_raw_text_to_inputs, \
    append_templates_to_inputs, append_mask_token_to_inputs, difference, check_if_bpe_tokenizer, replace_tokens_to_mask
from omneval.registry import register_processor
import pdb
import re

warnings.filterwarnings('ignore')


@register_processor('classification')
class ProcessorForClassification(BaseProcessor):

    def __init__(self, config):
        assert hasattr(config, 'templates'), "prompt templates should be specified."
        assert hasattr(config, 'label_mappings'), "label mappings should be specified."
        super(ProcessorForClassification, self).__init__(config)
        self.remove_punc = getattr(self.config, "remove_punc", False)

    @property
    def prompt_count(self):
        return len(self.config.templates)

    def prompt_schema(self, pid):
        return self.config.templates[pid]

    def generate_dataset(self, pid=0):
        """Prompting each instance and build dataset directly for the Evaluator"""
        self.labels_ids, self.labels_masks, self.mask_length = self.convert_verbalizers_to_ids(pid)
        prompt_schema = self.config.templates[pid]
        remove_columns = difference(self.raw_data.features.keys(), self.label_name)
        # generate a sample for calibration(no text but only prompt template)
        calibrate_word = self.generate_calibrate_example(pid)
        prompt_length = sum(calibrate_word['attention_mask'])
        return self.raw_data.map(
            lambda x: self.prompting(example=x,
                                     prompt_schema=prompt_schema,
                                     max_length=self.max_seq_length - prompt_length),
            remove_columns=remove_columns)

    def prompting(self, example, prompt_schema, max_length=512):
        text = ''
        text_length_cnt = 0
        for item in prompt_schema.split('|'):
            item = item.strip()
            # for the placeholders of label_tokens
            if item == '<mask>':
                text = append_mask_token_to_inputs(text, self.mask_token, self.mask_length)
            elif item in example and isinstance(example[item], str):
                appended_text = normalize_raw_text_to_inputs(example[item], self.remove_punc)
                appended_text, appended_length = truncate_text(appended_text, self.tokenizer, max_length-text_length_cnt)
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
                res['input_ids'] += self.mask_length * [self.mask_token_id, ]
                res['attention_mask'] += self.mask_length * [1]
                text_len += self.mask_length
            else:
                res['decoder_input_ids'] = self.tokenizer.encode(((self.mask_token + ' ') * self.mask_length).strip())
                mask_start_pos = res['decoder_input_ids'].index(self.mask_token_id)
        try:
            for i in range(mask_start_pos, mask_start_pos + self.mask_length):
                res['mask_pos'][i] = 1
        except:
            pdb.set_trace()
        ## TODO: this part may be wrong for other tasks
        res.update({self.label_name: example[self.label_name]})
        # res['input_ids'] += (self.max_seq_length - text_len) * [self.padding_id]
        # res['attention_mask'] += (self.max_seq_length - text_len) * [0]
        # if res.get('token_type_ids'):
        #     res['token_type_ids'] += (self.max_seq_length - text_len) * [0]
        # if res.get('decoder_input_ids'):
        #     res['decoder_input_ids'] += (self.max_seq_length - len(res.get('decoder_input_ids'))) * [0]
        return res

    def convert_verbalizers_to_ids(self, pid):
        """

        :return: [[[372], [205]], [[1099], [6587]]] like this
        """
        # create label token matrices
        mask_length = 0
        labels_ids = []
        for class_labels in self.get_label_tokens(pid):
            class_labels_ids = []
            for label in class_labels:
                token_ids = self.tokenizer.encode(label,  add_special_tokens=False)
                mask_length = max(mask_length, len(token_ids))
                class_labels_ids.append(token_ids)
            labels_ids.append(class_labels_ids)
        labels_masks = []
        for class_ids in labels_ids:
            class_masks = []
            for label in class_ids:
                label_mask = [1] * len(label) + [0] * (mask_length-len(label))
                label += [self.padding_id] * (mask_length - len(label))
                class_masks.append(label_mask)
            labels_masks.append(class_masks)
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
        candidate_idx = torch.LongTensor(self.labels_ids)
        candidate_idx_mask = torch.LongTensor(self.labels_masks)
        qa_prompting = getattr(self.config, "qa_prompting", None)
        calibrate_input = self.generate_calibrate_example(pid) if getattr(self.config, "calibrate", False) else None
        return {'candidate_idx': candidate_idx, 'candidate_idx_mask': candidate_idx_mask,
                'candidate_labels': candidate_labels, "qa_prompting": qa_prompting, "calibrate_input": calibrate_input}


