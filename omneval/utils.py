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
               'facebook/bart-large-cnn', 'sshleifer/distilbart-cnn-12-6']
T5_MODELS = ['t5-base', 't5-large']

# 2021-12-20
# BERT_MODELS = ['distilroberta-base',
#                'distilbert-base-uncased', 'distilbert-base-multilingual-cased', 'xlm-roberta-base',
#                'bert-base-multilingual-cased', 'albert-base-v2', 'vinai/bertweet-base',
#                'xlm-roberta-large', 'sentence-transformers/all-mpnet-base-v2',
#                'beomi/kcbert-base', 'albert-large-v2',
#                'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', 'johngiorgi/declutr-base',
#                'microsoft/mpnet-base', ]
#
# BERT_MODELS = ['albert-base-v1', 'sentence-transformers/multi-qa-distilbert-cos-v1',
#                'beomi/kcbert-large', 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
#                'xlm-mlm-en-2048', 'albert-xxlarge-v2', 'sshleifer/tiny-distilroberta-base',
#                'Davlan/bert-base-multilingual-cased-finetuned-amharic', 'dmis-lab/biobert-base-cased-v1.2',
#                 ]

# 2021-12-30
# BERT_MODELS = ['nlpaueb/legal-bert-base-uncased', 'emilyalsentzer/Bio_ClinicalBERT', 'google/bigbird-roberta-large',
#                'emilyalsentzer/Bio_Discharge_Summary_BERT', 'sentence-transformers/all-distilroberta-v1',
#                'cointegrated/rubert-tiny', 'saibo/legal-roberta-base', 'google/electra-small-generator',
#                'google/electra-base-generator', 'GroNLP/hateBERT', 'bioformers/bioformer-cased-v1.0', 'EMBEDDIA/crosloengual-bert',
#                'sentence-transformers/all-roberta-large-v1', 'albert-xxlarge-v1', 'kornosk/bert-political-election2020-twitter-mlm',
#                'climatebert/distilroberta-base-climate-f', 'zlucia/custom-legalbert', 'anferico/bert-for-patents'
#                ]

#
# BERT_MODELS = ['distilbert-base-uncased-finetuned-sst-2-english', 'bhadresh-savani/distilbert-base-uncased-emotion',
#                'yiyanghkust/finbert-tone', 'lordtt13/emo-mobilebert', 'finiteautomata/bertweet-base-sentiment-analysis',
#                'ProsusAI/finbert', 'typeform/distilbert-base-uncased-mnli', 'cross-encoder/nli-MiniLM2-L6-H768',
#                'siebert/sentiment-roberta-large-english', 'Narsil/deberta-large-mnli-zero-cls',
#                'mrm8488/codebert-base-finetuned-detect-insecure-code', 'microsoft/deberta-base-mnli',
#                'microsoft/deberta-large-mnli',
#                ]


#2021-01-01
# BERT_MODELS = ['deepset/roberta-base-squad2', 'cardiffnlp/twitter-roberta-base-sentiment',
#                'cardiffnlp/twitter-xlm-roberta-base', 'julien-c/bert-xsmall-dummy',
#                'bert-large-cased-whole-word-masking', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
#                'google/electra-large-generator', 'klue/bert-base', 'nlptown/bert-base-multilingual-uncased-sentiment',
#                'dslim/bert-base-NER', 'bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12',
#                'distilbert-base-cased-distilled-squad', 'distilbert-base-uncased-distilled-squad',
#                'mrm8488/bioclinicalBERT-finetuned-covid-papers', 'sentence-transformers/all-mpnet-base-v1',
#                'flax-sentence-embeddings/all_datasets_v3_mpnet-base','facebook/muppet-roberta-large',
#                'facebook/muppet-roberta-base', 'lordtt13/COVID-SciBERT', 'jhu-clsp/bibert-ende',
#                ]
# GPT_MODELS = ['distilgpt2']
# BART_MODELS = ['tuner007/pegasus_paraphrase']
# T5_MODELS = ['t5-small']

#2021-01-02
# BERT_MODELS = ['castorini/azbert-base', 'climatebert/distilroberta-base-climate-s', 'recobo/chemical-bert-uncased',
#               'albert-xlarge-v1', 'amine/bert-base-5lang-cased', 'ahmedrachid/FinancialBERT',
#               'flax-sentence-embeddings/all_datasets_v4_mpnet-base', 'climatebert/distilroberta-base-climate-d-s',
#               'EMBEDDIA/litlat-bert', 'Intel/distilbert-base-uncased-sparse-90-unstructured-pruneofa',
#               'nlp4good/psych-search', 'beatrice-portelli/DiLBERT', 'monsoon-nlp/muril-adapted-local',
#               'raynardj/pmc-med-bio-mlm-roberta-large', 'Geotrend/distilbert-base-en-cased',
#               'Intel/bert-base-uncased-sparse-70-unstructured', 'ayansinha/lic-class-scancode-bert-base-cased-L32-1',
#               'flax-sentence-embeddings/reddit_single-context_mpnet-base', ]
# GPT_MODELS = []
# BART_MODELS = ['ccdv/lsg-pegasus-large-4096', 'ccdv/lsg-bart-base-4096']
# T5_MODELS = []


# 2021-01-04
# BERT_MODELS = ['typeform/distilroberta-base-v2', 'Geotrend/bert-base-en-cased', 'jhu-clsp/roberta-large-eng-ara-128k',
#               'recobo/agriculture-bert-uncased', 'biu-nlp/cdlm', 'ayansinha/false-positives-scancode-bert-base-uncased-L8-1',
#               'zlucia/bert-double', 'ccdv/lsg-base-4096', 'mlcorelib/debertav2-base-uncased',
#               'abhi1nandy2/Bible-roberta-base', 'junnyu/electra_small_generator', 'mlcorelib/deberta-base-uncased',
#               'alexanderfalk/danbert-small-cased', 'ccdv/legal-lsg-base-uncased-4096', 'junnyu/roformer_small_generator',
#               'antoiloui/netbert', 'flax-sentence-embeddings/all_datasets_v3_distilroberta-base', 'vesteinn/XLMR-ENIS',
#               'byeongal/bert-base-uncased', 'raynardj/roberta-pubmed', 'wilsontam/bert-base-uncased-dstc9',
#               'Barytes/hellohf', 'Intel/bert-base-uncased-mnli-sparse-70-unstructured-no-classifier',
#               'Intel/distilbert-base-uncased-sparse-85-unstructured-pruneofa', 'ccdv/legal-lsg-small-uncased-4096',
#               'benyong/testmodel', 'sramasamy8/testModel'
#               ]
#
# GPT_MODELS = []
# BART_MODELS = []
# T5_MODELS = []

# 2021-01-07
# BERT_MODELS = ['typeform/distilroberta-base-v2', 'Geotrend/bert-base-en-cased', 'jhu-clsp/roberta-large-eng-ara-128k',
#               'recobo/agriculture-bert-uncased', 'biu-nlp/cdlm', 'ayansinha/false-positives-scancode-bert-base-uncased-L8-1',
#               'zlucia/bert-double', 'ccdv/lsg-base-4096', 'mlcorelib/debertav2-base-uncased',
#               'abhi1nandy2/Bible-roberta-base', 'junnyu/electra_small_generator', 'mlcorelib/deberta-base-uncased',
#               'alexanderfalk/danbert-small-cased', 'ccdv/legal-lsg-base-uncased-4096', 'junnyu/roformer_small_generator',
#               'antoiloui/netbert', 'flax-sentence-embeddings/all_datasets_v3_distilroberta-base', 'vesteinn/XLMR-ENIS',
#               'byeongal/bert-base-uncased', 'raynardj/roberta-pubmed', 'wilsontam/bert-base-uncased-dstc9',
#               'Barytes/hellohf', 'Intel/bert-base-uncased-mnli-sparse-70-unstructured-no-classifier',
#               'Intel/distilbert-base-uncased-sparse-85-unstructured-pruneofa', 'ccdv/legal-lsg-small-uncased-4096',
#               'benyong/testmodel', 'sramasamy8/testModel'
#               ]
#
# GPT_MODELS = []
# BART_MODELS = []
# T5_MODELS = []

def collate_fn(batch, exclude=[]):
    if callable(getattr(batch, "keys", None)):
        keys = batch.keys()
        return {k: torch.LongTensor([batch[k]]) if k not in exclude else [bz[k] for bz in batch] for k in keys}
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


def pad_input_ids(example, max_seq_length, padding_id, truncated='right'):
    text_len = len(example['input_ids'])
    # assert text_len <= max_seq_length, "length of input_ids %s, max_length thres %d"%(text_len, max_seq_length)
    if text_len > max_seq_length:
        # print("length of input_ids %s, max_length thres %d; trancate from %s"%(text_len, max_seq_length, truncated))
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
    filename = config.meta_prefix+'_'+config.task+'.json'
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
        print("Using template: %s"%template)
    for k, v in eval_results.items():
        print(k, ":", v)


def normalize_raw_text_to_inputs(text, remove_punc=False, lowercase=True):
    # TODO: Need to think that why we need to lowercase all words
    if lowercase:
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


def append_templates_to_inputs(text, templates, next_line=False):
    text = text.strip()
    if text and templates and templates[0] not in string.punctuation:
        text += (' ' if not next_line else '\n')
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

def adjust_length(config):
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
    return isinstance(tokenizer, GPT2Tokenizer) or isinstance(tokenizer, GPT2TokenizerFast)


def check_if_answers_single_tokens(label_mappings):
    BASE_MODEL = ['bert-base-uncased', 'roberta-base', 'openai-gpt', 'gpt2', 'facebook/bart-base', 't5-base']
    logging.info("Checking if length of all answers are 1 for most base models. \nBase models: %s"%str(BASE_MODEL))
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


# Utils for CONLL2003 NER

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
            chunk = (chunk_type, chunk_start, i-1)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i-1)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq)-1)
        chunks.append(chunk)

    return chunks
