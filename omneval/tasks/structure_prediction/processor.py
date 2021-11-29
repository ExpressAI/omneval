import torch
import warnings
from omneval.tasks import BaseProcessor
from omneval.utils import difference, pad_input_ids, normalize_raw_text_to_inputs, truncate_text, \
    append_templates_to_inputs, append_mask_token_to_inputs, check_if_bpe_tokenizer, make_sentence
from omneval.registry import register_processor
warnings.filterwarnings('ignore')



@register_processor('structure_prediction')
class ProcessorForStructurePrediction(BaseProcessor):
    def __init__(self, config):
        assert hasattr(config, 'templates'), "prompt templates should be specified."
        assert hasattr(config, 'label_mappings'), "label mappings should be specified."
        super(ProcessorForStructurePrediction, self).__init__(config)
        self.labels_ids, self.labels_masks, self.mask_length = self.convert_verbalizers_to_ids()
        self.remove_punc = getattr(self.config, "remove_punc", False)

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
                                     max_length=self.max_seq_length-prompt_length,
                                     sentence_label=self.config.sentence_label),
                                     remove_columns=remove_columns,)

    def prompting(self, example, prompt_schema, max_length=512, sentence_label='sentence'):
        text = ''
        # TODO: Should check if it can be generalized to all RE datasets
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
                # If example[item] is not the beginning of the sentence, lowercase all words
                appended_text = normalize_raw_text_to_inputs(appended_text, self.config.remove_punc)
                appended_text, appended_length = truncate_text(appended_text, self.tokenizer, max_length-text_length_cnt)
                text_length_cnt += appended_length
                text += appended_text
            # for entities
            elif item == '<e>':
                text = text.strip()
                if text:
                    text += ' '
                text += example['token']
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
        # create label token matrices， if answers are of different length, we pad answer tokens to the same length and
        # use labels_masks to mask padded position(exclude them in probability/ppl calculation)
        mask_length = [0, 0]
        labels_ids = []
        for class_labels in self.get_label_tokens():
            class_labels_ids = []
            for idx, label in enumerate(class_labels):
                token_ids = self.tokenizer.encode(label,  add_special_tokens=False)
                mask_length[idx] = max(mask_length[idx], len(token_ids))
                class_labels_ids.append(token_ids)
            labels_ids.append(class_labels_ids)
        labels_masks = []
        for class_ids in labels_ids:
            class_masks = []
            for idx, label in enumerate(class_ids):
                label_mask = [1] * len(label) + [0] * (mask_length[idx]-len(label))
                label += [self.padding_id] * (mask_length[idx] - len(label))
                class_masks.append(label_mask)
            labels_masks.append(class_masks)
        return labels_ids, labels_masks, mask_length

    def get_label_tokens(self):
        # For GPT2 tokenizers, we should add a space before each word.
        if check_if_bpe_tokenizer(self.tokenizer):
            label_tokens = [[' '+l.lower() for l in labels] for labels in self.config.label_mappings]
        else:
            label_tokens = self.config.label_mappings
        return label_tokens

    def generate_aux_inputs(self, pid=0):
        """ generate auxiliary input other than datasets"""
        candidate_labels = self.config.labels
        candidate_idx = torch.LongTensor(self.labels_ids)
        candidate_idx_mask = torch.LongTensor(self.labels_masks)
        mask_length = self.mask_length
        qa_prompting = getattr(self.config, "qa_prompting", None)
        calibrate_input = self.generate_calibrate_example(pid) if getattr(self.config, "calibrate", False) else None
        return {'candidate_idx': candidate_idx, 'candidate_idx_mask': candidate_idx_mask,
                'candidate_labels': candidate_labels, "qa_prompting": qa_prompting, "calibrate_input": calibrate_input,
                'mask_length': sum(mask_length)}
