import torch
import pdb
import json
import os
import string

def collate_fn(batch, exclude=[]):
    if callable(getattr(batch, "keys", None)):
        keys = batch.keys()
        return {k: torch.LongTensor([batch[k]]) for k in keys}
    else:
        keys = batch[0].keys()
        return {k: (torch.LongTensor([bz[k] for bz in batch]) if k not in exclude else [bz[k] for bz in batch]) for k in
                keys}


def get_logits(outputs):
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    elif hasattr(outputs, 'prediction_logits'):
        logits = outputs.prediction_logits
    else:
        raise NotImplementedError
    return logits


def merge_fn(dict1, dict2):
    for k, v in dict2.items():
        dict1[k] += v


def difference(list1, list2):
    return [item for item in list1 if item not in list2]


def truncate_text(text, tokenizer, target_length):
    while True:
        tokenized_text = tokenizer.encode(text, max_length=target_length, add_special_tokens=False)
        text = tokenizer.decode(tokenized_text[: target_length], clean_up_tokenization_spaces=True, skip_special_tokens=True)
        text_length = len(tokenizer.encode(text, add_special_tokens=False))
        if text_length <= target_length:
            break

    return text, text_length


def pad_input_ids(example, max_seq_length, padding_id):
    text_len = len(example['input_ids'])
    assert text_len <= max_seq_length, "length of input_ids %s, max_length thres %d"%(text_len, max_seq_length)
    # if text_len > max_seq_length:
    #     for k in example.keys():
    #         if truncated == 'right':
    #             example[k] = example[k][: max_seq_length]
    #         else:
    #             example[k] = example[k][-max_seq_length:]
    example['input_ids'] += (max_seq_length - text_len) * [padding_id]
    example['attention_mask'] += (max_seq_length - text_len) * [0]
    if example.get('token_type_ids'):
        example['token_type_ids'] += (max_seq_length - text_len) * [0]
    return example


def init_eval_result(config):
    meta = {'task': config.task,
            'task_type': config.task_type,
            'datasets': config.dataset_name if isinstance(config.dataset_name, str) else '/'.join(config.dataset_name),
            'prompts': []}
    for template in config.templates:
        meta_template = dict()
        meta_template['template'] = template
        meta_template['setting'] = 'zero-shot'
        meta_template['results'] = []
        meta['prompts'].append(meta_template)
    if meta['task_type'] == 'classification':
        meta['supported_plms'] = ["masked_lm", "left_to_right", "encoder_decoder"]
        meta['answers'] = dict(zip(config.labels, config.label_mappings))
    return meta


def write_meta_eval_to_json(output, config):
    filename = 'meta_'+config.task+'.json'
    filename = os.path.join(config.out_dir, filename)
    with open(filename, 'w') as f:
        json.dump(output, f, indent=1)


def print_eval_result(eval_results, template=None):
    if template:
        print("Using template: %s"%template)
    for k, v in eval_results.items():
        print(k, ":", v)


def normalize_raw_text_to_inputs(text, remove_punc=False):
    text = text.strip().lower()
    # if cur_text and cur_text.strip()[-1] not in string.punctuation:
    #     append_text = append_text.lower()
    # else:
    #     append_text = append_text.capitalize()
    if not text:
        return text
    if text[-1] in string.punctuation:
        if not remove_punc:
            text = text[: -1].strip() + text[-1]
        else:
            text = text[: -1].strip()
    else:
        if not remove_punc:
            text += '.'
    return text


def append_templates_to_inputs(text, templates):
    text = text.strip()
    if text and templates not in string.punctuation:
        text += ' '
    text += templates
    if text and text[-1] != ' ':
        text += ' '
    return text


def append_mask_token_to_inputs(text, mask_token, mask_length):
    # TODO: Should find better adaptation ways for different tokenizers(GPT2)
    # for raw inputs
    if mask_token != '<|endoftext|>':
        text = text.strip()
        if text:
            text += ' '
        text += (mask_token + ' ') * mask_length
    else:
        text += mask_token * mask_length
    return text