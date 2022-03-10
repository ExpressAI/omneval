import torch
import json
import os
import string
from transformers import AutoConfig, AutoTokenizer, GPT2Tokenizer, GPT2TokenizerFast
import logging
import collections

BERT_MODELS = ['bert-base-uncased', 'roberta-base', 'bert-large-uncased', 'roberta-large', 'distilroberta-base',
               'distilbert-base-uncased']

GPT_MODELS = ['openai-gpt', 'gpt2']

BART_MODELS = ['facebook/bart-base', 'facebook/bart-large',
               'facebook/bart-large-cnn', 'sshleifer/distilbart-cnn-12-6', 'facebook/bart-large-mnli',
               'textattack/facebook-bart-large-SST-2', 'facebook/bart-large-xsum']

T5_MODELS = ['t5-base', 't5-large']

SUPPORTED_MODELS = {
    'bert': BERT_MODELS,
    'bart': BART_MODELS,
    'gpt': GPT_MODELS,
    't5': T5_MODELS
}


def collate_fn(batch, exclude=[]):
    """make batches for the data"""
    if callable(getattr(batch, "keys", None)):
        keys = batch.keys()
        return {k: torch.LongTensor([batch[k]]) if k not in exclude else [bz[k] for bz in batch] for k in keys}
    else:
        keys = batch[0].keys()
        return {k: (torch.LongTensor([bz[k] for bz in batch]) if k not in exclude else [bz[k] for bz in batch]) for k in
                keys}


def get_logits(outputs):
    """get logits from the PLM"""
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
    """Function to truncate text into target length"""
    # direct truncation from the tokenized list may lead to an unwanted length after a second time tokenization.
    # So currently use a loop to make sure the text can be tokenized into the target length
    while True:
        tokenized_text = tokenizer.encode(text, max_length=target_length, add_special_tokens=False)
        text = tokenizer.decode(tokenized_text[: target_length], clean_up_tokenization_spaces=True,
                                skip_special_tokens=True)
        text_length = len(tokenizer.encode(text, add_special_tokens=False))
        if text_length <= target_length:
            break

    return text, text_length


def pad_input_ids(example, max_seq_length, padding_id, truncated='right'):
    """Pad inputs"""
    text_len = len(example['input_ids'])
    if text_len > max_seq_length:
        for k in example.keys():
            if truncated == 'right':
                example[k] = example[k][: max_seq_length]
            else:
                example[k] = example[k][-max_seq_length:]
    example['input_ids'] += (max_seq_length - text_len) * [padding_id]
    example['attention_mask'] += (max_seq_length - text_len) * [0]
    if example.get('token_type_ids'):
        example['token_type_ids'] += (max_seq_length - text_len) * [0]
    return example


def init_eval_result(config):
    """Initialize the evaluation fields"""
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
    """write meta evaluation metrics into json file"""
    filename = config.meta_prefix + '_' + config.task + '.json'
    filename = os.path.join(config.out_dir, filename)
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            meta_old = json.load(f)
        meta = merge_meta_result(meta_old, output)
    else:
        meta = output
    with open(filename, 'w') as f:
        json.dump(meta, f, indent=1)


def merge_meta_result(output1, output2):
    """Merge results from existing result files"""
    t_dict = get_temp_result_dict(output1)
    for item in output2['prompts']:
        if item['template'] not in t_dict:
            t_dict[item['template']] = dict()
        for result in item['results']:
            t_dict[item['template']][result['plm']] = result
    output1['prompts'] = reverse_to_json(t_dict, setting='zero-shot')
    return output1


def reverse_to_json(t_dict, setting='zero-shot'):
    res = []
    for template, result in t_dict.items():
        res.append(
            {
                'template': template,
                'setting': setting,
                'results': list(result.values())
            }
        )
    return res


def get_temp_result_dict(output):
    return {item['template']: {res['plm']: res for res in item['results']} for item in output['prompts']}


def print_eval_result(eval_results, template=None):
    if template:
        print("Using template: %s" % template)
    for k, v in eval_results.items():
        print(k, ":", v)


def normalize_raw_text_to_inputs(text, remove_punc=False, lowercase=True):
    """function to normalize all """
    # Currently we lowercase all input texts as normalization
    if lowercase:
        text = text.strip().lower()
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


def append_templates_to_inputs(text, templates, next_line=False):
    """Add prompt template to the original input, should care about whitespaces and punctuations"""
    text = text.strip()
    if text and templates and templates[0] not in string.punctuation:
        text += (' ' if not next_line else '\n')
    text += templates
    if text and text[-1] != ' ':
        text += ' '
    return text


def append_mask_token_to_inputs(text, mask_token, mask_length):
    """ Add mask tokens for MLM classification"""
    # special for GPT2
    if mask_token != '<|endoftext|>':
        text = text.strip()
        if text:
            text += ' '
        text += (mask_token + ' ') * mask_length
    else:
        text += mask_token * mask_length
    return text


def adjust_length(config):
    """Return the expected length for non-prompt text"""
    model_config = AutoConfig.from_pretrained(config.arch)
    max_position_embeddings = getattr(model_config, 'max_position_embeddings', config.max_seq_length)
    max_seq_length = min(config.max_seq_length, max_position_embeddings)
    decode_seq_length = getattr(config, 'decode_max_length', 0)
    if config.arch in GPT_MODELS:
        if max_position_embeddings >= max_seq_length + decode_seq_length:
            return max_seq_length
        else:
            return max_position_embeddings - decode_seq_length
    else:
        return min(max_position_embeddings, max_seq_length)


def get_masked_tokens(config, tokenizer):
    """get masked token representation for PLM"""
    if not tokenizer._mask_token:
        if config.arch.startswith('t5'):
            mask_token = '<extra_id_0>'
        else:
            mask_token = '<mask>'
            # TODO: Check if this part works
            tokenizer.add_tokens([mask_token])
    else:
        mask_token = tokenizer.mask_token
    mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
    return mask_token, mask_token_id


def check_if_bpe_tokenizer(tokenizer):
    """ check whether PLM uses BPE tokenizers"""
    return isinstance(tokenizer, GPT2Tokenizer) or isinstance(tokenizer, GPT2TokenizerFast)


def check_if_answers_single_tokens(label_mappings):
    """For classification task, check if the answer candidate is single-token"""
    BASE_MODEL = ['bert-base-uncased', 'roberta-base', 'openai-gpt', 'gpt2', 'facebook/bart-base', 't5-base']
    logging.info("Checking if length of all answers are 1 for most base models. \nBase models: %s" % str(BASE_MODEL))
    base_tokenizers = [AutoTokenizer.from_pretrained(model) for model in BASE_MODEL]
    memo = collections.defaultdict(set)
    res = True
    for idx, tokenizer in enumerate(base_tokenizers):
        for label_mapping in label_mappings:
            if check_if_bpe_tokenizer(tokenizer):
                label_tokens = [[' ' + l.lower() for l in labels] for labels in label_mapping]
            else:
                label_tokens = label_mapping
            for labels in label_tokens:
                for label in labels:
                    tokenized_tokens = tokenizer.tokenize(label.lower())
                    if len(tokenized_tokens) > 1:
                        res = False
                        memo[BASE_MODEL[idx]].add(label)
    if res:
        logging.info("All manual answers are single token, Great")
    else:
        logging.warning("We find several answers multi-tokens, the different length of answers may lead to "
                        "reduced zero-shot performance for PLMs, They are")
        logging.warning(memo)
    return res


def make_sentence(l):
    """add string lists to a sentence"""
    text = ''
    for idx, token in enumerate(l):
        if token not in string.punctuation:
            text += ' '
        text += token.lower()
    return text.strip()


def replace_tokens_to_mask(text, replace_token, mask_token):
    if text == '':
        return text
    if replace_token in text:
        text = text.replace(replace_token, mask_token, 1)
    else:
        # if masked tokens is truncated
        if text[-1] in string.punctuation:
            text = text[:-1].strip() + ' ' + mask_token + text[-1]
        else:
            text = text.strip() + ' ' + mask_token
    return text


"""
Below are functions for CONLL2003 NER metrics
"""


def get_chunk_type(tok):
    """
	Args:
		tok: id of token, ex 4
		idx_to_tag: dictionary {4: "B-PER", ...}
	Returns:
		tuple: "B", "PER"
	"""
    # tag_name = idx_to_tag[tok]
    tag_class = tok.split('-')[0]
    # tag_type = tok.split('-')[-1]
    tag_type = '-'.join(tok.split('-')[1:])
    return tag_class, tag_type


def get_entity_span_ids(seq, tags, default='O'):
    """
	tags:dic{'per':1,....}
	Args:
		seq: [4, 4, 0, 0, ...] sequence of labels
		tags: dict["O"] = 4
	Returns:
		list of (chunk_type, chunk_start, chunk_end)

	Example:
		seq = [4, 5, 0, 3]
		tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
		result = [("PER", 0, 2), ("LOC", 3, 4)]
	"""
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        tok = idx_to_tag[tok]
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i - 1)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i - 1)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq) - 1)
        chunks.append(chunk)

    return chunks
