import torch
import warnings
from .. import register_processor, difference
from .. import BaseProcessor
import logging
import pdb
from transformers import GPT2Tokenizer, GPT2TokenizerFast

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

    def generate_dataset(self, pid=0):
        padding_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        order_t, order_l = pid // len(self.config.label_mappings), pid % len(
            self.config.label_mappings)
        prompt_schema, label_tokens = self.config.templates[order_t], self.get_label_tokens(order_l)
        info = "Using %s, and label mappings %s"%(prompt_schema, str([l.strip() for l in label_tokens]))
        logging.info(info)
        mask_length = len(self.tokenizer.tokenize(label_tokens[0], add_special_tokens=False))
        label_name = getattr(self.config, 'label_name', 'label')
        remove_columns = difference(self.raw_data.features.keys(), ['label'])
        for l in label_tokens:
            if mask_length != len(self.tokenizer.tokenize(l, add_special_tokens=False)):
                raise ValueError("Currently the framework only supports label mappings of the same token sizes")

        def prompting(example, tokenizer, prompt_schema, mask_length, max_seq_length, padding_id=0):
            text = ''
            mask_token = tokenizer.mask_token if tokenizer._mask_token is not None else tokenizer.unk_token
            mask_token_id = tokenizer.mask_token_id if tokenizer._mask_token is not None else tokenizer.unk_token_id
            remove_punc = getattr(self.config, "remove_punc", False)

            for item in prompt_schema.split('|'):
                item = item.strip()
                # for the placeholders of label_tokens
                if item == '<mask>':
                    # TODO: Should find better adaptation ways for different tokenizers(GPT2)
                    if mask_token != '<|endoftext|>':
                        text = text.strip()
                        if text:
                            text += ' '
                        text += (mask_token + ' ') * mask_length
                    else:
                        text += mask_token * mask_length
                    # for raw inputs
                elif example.get(item) and isinstance(example[item], str):
                    tmp = example[item].strip()
                    if text and text.strip()[-1] not in '.?!':
                        tmp = tmp.lower()
                    if tmp[-1] in '.,?!':
                        text += tmp[: -1].strip()
                        if not remove_punc:
                            text += tmp[-1]
                    else:
                        text += tmp
                        if not remove_punc:
                            text += '.'
                # for prompting templates
                else:
                    text = text.strip()
                    if text and item not in '.,?!':
                        text += ' '
                    text += item
                    if text and text[-1] != ' ':
                        text += ' '
            res = tokenizer(text.strip())
            text_len = len(res['input_ids'])
            res['mask_pos'] = [0] * max_seq_length
            # If no mask in the template, append mask in the end
            try:
                mask_start_pos = res['input_ids'].index(mask_token_id)
            except:
                mask_start_pos = 0
                if not getattr(self.config, 'qa_prompting', False):
                    res['input_ids'] += mask_length * [mask_token_id, ]
                    res['attention_mask'] += mask_length * [1]
                    text_len += mask_length
            for i in range(mask_start_pos, mask_start_pos + mask_length):
                res['mask_pos'][i] = 1
            ## TODO: this part may be wrong for other tasks
            res.update({'label': example[label_name]})
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
            remove_columns=remove_columns)

    def convert_verbalizers_to_ids(self, idx):
        return [list(self.tokenizer.encode(x,  add_special_tokens=False)) for x in self.get_label_tokens(idx)]

    def get_label_tokens(self, idx):
        # For GPT2 tokenizers, we should add a space before each word.
        if isinstance(self.tokenizer, GPT2Tokenizer) or isinstance(self.tokenizer, GPT2TokenizerFast):
            label_tokens = [' '+l for l in self.config.label_mappings[idx]]
        else:
            label_tokens = self.config.label_mappings[idx]
        return label_tokens


    def generate_aux_inputs(self, pid=0):
        order_t, order_l = pid // len(self.config.label_mappings), pid % len(
            self.config.label_mappings)
        candidate_idx = torch.LongTensor(self.convert_verbalizers_to_ids(order_l))
        candidate_labels = self.config.labels
        qa_prompting = getattr(self.config, "qa_prompting", None)
        return {'candidate_idx': candidate_idx, 'candidate_labels': candidate_labels, "qa_prompting": qa_prompting}


@register_processor('classification_m')
class ProcessorForClassificationMultipleAnswer(BaseProcessor):

    def __init__(self, arch, config):
        assert hasattr(config, 'templates'), "prompt templates should be specified."
        assert hasattr(config, 'label_mappings'), "label mappings should be specified."
        super(ProcessorForClassificationMultipleAnswer, self).__init__(arch, config)
        self.label_mapping = self.convert_verbalizers_to_ids()
        for labels in self.label_mapping:
            for l in labels:
                if len(l) != 1:
                    labels.remove(l)
                    logging.info("Currently this processor only supports single token labels, %s omitted"%(self.tokenizer.decode(l)))

    @property
    def prompt_count(self):
        return len(self.config.templates)

    def generate_dataset(self, pid=0):
        padding_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        prompt_schema = self.config.templates[pid]
        mask_length = 1
        label_name = getattr(self.config, 'label_name', 'label')
        remove_columns = difference(self.raw_data.features.keys(), ['label'])
        def prompting(example, tokenizer, prompt_schema, mask_length, max_seq_length, padding_id=0):
            text = ''
            mask_token = tokenizer.mask_token if tokenizer._mask_token is not None else tokenizer.unk_token
            mask_token_id = tokenizer.mask_token_id if tokenizer._mask_token is not None else tokenizer.unk_token_id
            remove_punc = getattr(self.config, "remove_punc", False)

            for item in prompt_schema.split('|'):
                item = item.strip()
                # for the placeholders of label_tokens
                if item == '<mask>':
                    # TODO: Should find better adaptation ways for different tokenizers(GPT2)
                    if mask_token != '<|endoftext|>':
                        text = text.strip()
                        if text:
                            text += ' '
                        text += (mask_token + ' ') * mask_length
                    else:
                        text += mask_token * mask_length
                    # for raw inputs
                elif example.get(item) and isinstance(example[item], str):
                    tmp = example[item].strip()
                    if text and text.strip()[-1] not in '.?!':
                        tmp = tmp.lower()
                    if tmp[-1] in '.,?!':
                        text += tmp[: -1].strip()
                        if not remove_punc:
                            text += tmp[-1]
                    else:
                        text += tmp
                        if not remove_punc:
                            text += '.'
                # for prompting templates
                else:
                    text = text.strip()
                    if text and item not in '.,?!':
                        text += ' '
                    text += item
                    if text and text[-1] != ' ':
                        text += ' '
            res = tokenizer(text.strip())
            text_len = len(res['input_ids'])
            res['mask_pos'] = [0] * max_seq_length
            # If no mask in the template, append mask in the end
            try:
                mask_start_pos = res['input_ids'].index(mask_token_id)
            except:
                mask_start_pos = 0
                if not getattr(self.config, 'qa_prompting', False):
                    res['input_ids'] += mask_length * [mask_token_id, ]
                    res['attention_mask'] += mask_length * [1]
                    text_len += mask_length
            for i in range(mask_start_pos, mask_start_pos + mask_length):
                res['mask_pos'][i] = 1
            ## TODO: this part may be wrong for other tasks
            res.update({'label': example[label_name]})
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
            remove_columns=remove_columns)

    def convert_verbalizers_to_ids(self):
        """

        :return: [[[372], [205]], [[1099], [6587]]] like this
        """
        return [list(self.tokenizer.encode(x,  add_special_tokens=False) for x in labels) for labels in self.get_label_tokens()]

    def get_label_tokens(self):
        # For GPT2 tokenizers, we should add a space before each word.
        if isinstance(self.tokenizer, GPT2Tokenizer) or isinstance(self.tokenizer, GPT2TokenizerFast):
            label_tokens = [[' '+l.lower() for l in labels] for labels in self.config.label_mappings]
        else:
            label_tokens = self.config.label_mappings
        return label_tokens


    def generate_aux_inputs(self, pid=0):
        candidate_labels = self.config.labels
        candidate_idx = torch.LongTensor(self.label_mapping).squeeze(-1)
        qa_prompting = getattr(self.config, "qa_prompting", None)
        return {'candidate_idx': candidate_idx, 'candidate_labels': candidate_labels, "qa_prompting": qa_prompting}



@register_processor('classification_demo')
class ProcessorForClassificationDemo(BaseProcessor):

    def __init__(self, config):
        assert hasattr(config, 'templates'), "prompt templates should be specified."
        assert hasattr(config, 'label_mappings'), "label mappings should be specified."
        super(ProcessorForClassificationDemo, self).__init__(config)
        self.labels_ids, self.labels_masks, self.mask_length = self.convert_verbalizers_to_ids()
        self.remove_punc = getattr(self.config, "remove_punc", False)

    @property
    def prompt_count(self):
        return len(self.config.templates)

    def prompt_schema(self, pid):
        return self.config.templates[pid]

    def prompting(self, example, prompt_schema):
        text = ''

        for item in prompt_schema.split('|'):
            item = item.strip()
            # for the placeholders of label_tokens
            if item == '<mask>':
                # TODO: Should find better adaptation ways for different tokenizers(GPT2)
                if self.mask_token != '<|endoftext|>':
                    text = text.strip()
                    if text:
                        text += ' '
                    text += (self.mask_token + ' ') * self.mask_length
                else:
                    text += self.mask_token * self.mask_length
                # for raw inputs
            elif example.get(item) and isinstance(example[item], str):
                tmp = example[item].strip()
                if text and text.strip()[-1] not in '.?!':
                    tmp = tmp.lower()
                if tmp[-1] in '.,?!':
                    text += tmp[: -1].strip()
                    if not self.remove_punc:
                        text += tmp[-1]
                else:
                    text += tmp
                    if not self.remove_punc:
                        text += '.'
            # for prompting templates
            else:
                text = text.strip()
                if text and item not in '.,?!':
                    text += ' '
                text += item
                if text and text[-1] != ' ':
                    text += ' '
        res = self.tokenizer(text.strip())
        text_len = len(res['input_ids'])
        res['mask_pos'] = [0] * self.max_seq_length
        # If no mask in the template, append mask in the end
        try:
            mask_start_pos = res['input_ids'].index(self.mask_token_id)
        except:
            mask_start_pos = 0
            if not getattr(self.config, 'qa_prompting', False):
                res['input_ids'] += self.mask_length * [self.mask_token_id, ]
                res['attention_mask'] += self.mask_length * [1]
                text_len += self.mask_length
        for i in range(mask_start_pos, mask_start_pos + self.mask_length):
            res['mask_pos'][i] = 1
        ## TODO: this part may be wrong for other tasks
        res.update({self.label_name: example[self.label_name]})
        res['input_ids'] += (self.max_seq_length - text_len) * [self.padding_id]
        res['attention_mask'] += (self.max_seq_length - text_len) * [0]
        if res.get('token_type_ids'):
            res['token_type_ids'] += (self.max_seq_length - text_len) * [0]
        return res

    def convert_verbalizers_to_ids(self):
        """

        :return: [[[372], [205]], [[1099], [6587]]] like this
        """
        # create label token matrices
        mask_length = 0
        labels_ids = []
        for class_labels in self.get_label_tokens():
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
        qa_prompting = getattr(self.config, "qa_prompting", None)
        calibrate_input = self.generate_calibrate_example(pid) if getattr(self.config, "calibrate", False) else None
        return {'candidate_idx': candidate_idx, 'candidate_idx_mask': candidate_idx_mask,
                'candidate_labels': candidate_labels, "qa_prompting": qa_prompting, "calibrate_input": calibrate_input}

    def generate_calibrate_example(self, pid):
        prompt_schema = self.prompt_schema(pid)
        text = ''
        for item in prompt_schema.split('|'):
            item = item.strip()
            # for the placeholders of label_tokens
            if item == '<mask>':
                # TODO: Should find better adaptation ways for different tokenizers(GPT2)
                if self.mask_token != '<|endoftext|>':
                    text = text.strip()
                    if text:
                        text += ' '
                    text += (self.mask_token + ' ') * self.mask_length
                else:
                    text += self.mask_token * self.mask_length
                # for raw inputs
            elif item not in self.raw_data.features.keys():
                text = text.strip()
                if text and item not in '.,?!':
                    text += ' '
                text += item
                if text and text[-1] != ' ':
                    text += ' '

        res = self.tokenizer(text.strip())
        text_len = len(res['input_ids'])
        res['mask_pos'] = [0] * self.max_seq_length
        # If no mask in the template, append mask in the end
        try:
            mask_start_pos = res['input_ids'].index(self.mask_token_id)
        except:
            mask_start_pos = 0
            if not getattr(self.config, 'qa_prompting', False):
                res['input_ids'] += self.mask_length * [self.mask_token_id, ]
                res['attention_mask'] += self.mask_length * [1]
                text_len += self.mask_length
        for i in range(mask_start_pos, mask_start_pos + self.mask_length):
            res['mask_pos'][i] = 1
        res['input_ids'] += (self.max_seq_length - text_len) * [self.padding_id]
        res['attention_mask'] += (self.max_seq_length - text_len) * [0]
        if res.get('token_type_ids'):
            res['token_type_ids'] += (self.max_seq_length - text_len) * [0]
        return res


