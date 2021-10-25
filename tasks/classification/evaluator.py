import numpy as np
import torch
from torch.utils.data import DataLoader
from .. import BaseEvaluator, register_evaluator, get_logits
from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from sklearn.metrics import accuracy_score
import pdb
from torch.nn.functional import cross_entropy
import logging
BERT_MODELS = ['bert-base-uncased', 'roberta-base', 'bert-large-uncased', 'roberta-large', 'distilroberta-base',
               'distilbert-base-uncased']
GPT_MODELS = ['openai-gpt', 'gpt2']
BART_MODELS = ['facebook/bart-base', 'google/bert_for_seq_generation_L-24_bbc_encoder', 'facebook/bart-large',
               't5-base']
import re
def collate_fn(batch):
    if callable(getattr(batch, "keys", None)):
        keys = batch.keys()
        return {k: torch.LongTensor([batch[k]]) for k in keys}
    else:
        keys = batch[0].keys()
        return {k: torch.LongTensor([bz[k] for bz in batch]) for k in keys}


@register_evaluator('classification', BERT_MODELS)
class BERTEvaluatorForClassification(BaseEvaluator):

    def build_model(self, arch):
        return AutoModelForPreTraining.from_pretrained(arch).to(self.device)

    def preprocessing(self, dataset, **kwargs):
        kwargs['candidate_idx'] = kwargs.get('candidate_idx').to(self.device)
        kwargs['mask_length'] = kwargs['candidate_idx'].shape[-1]
        return dataset, kwargs


    def decode(self, batch, **kwargs):
        mask_pos = batch.pop('mask_pos')
        candidate_idx = kwargs.get('candidate_idx')
        candidate_labels = kwargs.get('candidate_labels')
        mask_length = kwargs.get('mask_length')
        with torch.no_grad():
            outputs = self.model(**batch)
        logits = get_logits(outputs)
        mask_logits = logits[mask_pos > 0].view(-1, mask_length, logits.shape[-1])
        candidate_logits = torch.stack(
            [mask_logits[:, i, :].index_select(-1, candidate_idx[:, i]) for i in range(mask_length)], axis=-1)
        candidate_logits = torch.nn.functional.log_softmax(candidate_logits, dim=1)
        candidate_logits = torch.sum(candidate_logits, axis=-1)
        return [candidate_labels[i] for i in candidate_logits.argmax(-1).cpu().detach().numpy()]


@register_evaluator('classification', GPT_MODELS)
class GPTEvaluatorForClassification(BaseEvaluator):

    def build_model(self, arch):
        return AutoModelForCausalLM.from_pretrained(arch).to(self.device)

    def preprocessing(self, dataset, **kwargs):
        kwargs['candidate_idx'] = kwargs.get('candidate_idx').to(self.device)
        kwargs['mask_length'] = kwargs['candidate_idx'].shape[-1]
        return dataset, kwargs

    def decode(self, batch, **kwargs):
        mask_pos = batch.pop('mask_pos')
        candidate_idx = kwargs.get('candidate_idx')
        candidate_labels = kwargs.get('candidate_labels')
        mask_length = kwargs.get('mask_length')
        padding_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        ppls = []
        with torch.no_grad():
            for i in range(len(candidate_labels)):
                batch['input_ids'][mask_pos > 0] = candidate_idx[i].repeat(batch['input_ids'].shape[0])
                tgt_ids = batch['input_ids'].clone()
                tgt_ids[tgt_ids == padding_id] = -100
                outputs = self.model(**batch, labels=tgt_ids)
                logits = get_logits(outputs)
                # logits->[seq_length, vocab_size, batch_size], ids ->[seq_length, batch_size]
                shifted_logits = logits[..., :-1, :]
                shifted_tgt_ids = tgt_ids[..., 1:]
                loss = cross_entropy(shifted_logits.permute(1, 2, 0), shifted_tgt_ids.permute(1, 0), reduce=False,
                                     ignore_index=-100)
                ppls.append(torch.sum(loss, axis=0) / torch.sum(~shifted_tgt_ids.eq(-100), axis=-1))
            ppls = torch.stack(ppls).transpose(1, 0)
        return [candidate_labels[i] for i in ppls.argmin(-1).cpu().detach().numpy()]


@register_evaluator('classification', BART_MODELS)
class BARTEvaluatorForClassification(BaseEvaluator):

    def build_model(self, arch):
        return AutoModelForSeq2SeqLM.from_pretrained(arch).to(self.device)

    def preprocessing(self, dataset, **kwargs):
        kwargs['candidate_idx'] = kwargs.get('candidate_idx').to(self.device)
        kwargs['mask_length'] = kwargs['candidate_idx'].shape[-1]
        return dataset, kwargs

    def decode(self, batch, **kwargs):
        candidate_idx = kwargs.get('candidate_idx')
        candidate_labels = kwargs.get('candidate_labels')
        mask_length = kwargs.get('mask_length')
        mask_pos = batch.pop('mask_pos')
        with torch.no_grad():
            outputs = self.model.generate(**batch)
        logits = get_logits(outputs)
        mask_logits = logits[mask_pos > 0].view(-1, mask_length, logits.shape[-1])
        candidate_logits = torch.stack(
            [mask_logits[:, i, :].index_select(-1, candidate_idx[:, i]) for i in range(mask_length)], axis=-1)
        candidate_logits = torch.nn.functional.log_softmax(candidate_logits, dim=1)
        candidate_logits = torch.sum(candidate_logits, axis=-1)
        return [candidate_labels[i] for i in candidate_logits.argmax(-1).cpu().detach().numpy()]

@register_evaluator('classification_m', BERT_MODELS)
class BERTEvaluatorForClassification(BaseEvaluator):

    def build_model(self):
        return AutoModelForPreTraining.from_pretrained(self.config.arch).to(self.device)

    def preprocessing(self, dataset, **kwargs):
        kwargs['candidate_idx'] = kwargs.get('candidate_idx').to(self.device)
        kwargs['mask_length'] = kwargs['candidate_idx'].shape[-1]
        kwargs['topk'] = getattr(self.config, 'topk', 3)
        return dataset, kwargs


    def decode(self, batch, **kwargs):
        mask_pos = batch.pop('mask_pos')
        candidate_idx = kwargs.get('candidate_idx')
        candidate_labels = kwargs.get('candidate_labels')
        mask_length = 1
        topk = kwargs.get('topk')
        with torch.no_grad():
            outputs = self.model(**batch)
        logits = get_logits(outputs)
        mask_logits = logits[mask_pos > 0].view(-1, logits.shape[-1])
        candidate_logits = torch.cat(
            [mask_logits.index_select(-1, candidate_idx[i]) for i in range(candidate_idx.shape[0])], dim=-1)
        max_tokens, max_indices = torch.topk(candidate_logits, k=topk)
        max_label_idx, max_cand_idx = max_indices // candidate_idx.shape[1], max_indices % candidate_idx.shape[1]
        res = {
            'predictions': [candidate_labels[i] for i in torch.mode(max_label_idx, -1)[0].cpu().detach().numpy()],
            'topk_tokens': [self.tokenizer.convert_ids_to_tokens(item) for item in candidate_idx[max_label_idx, max_cand_idx].cpu().detach().tolist()],
            'inputs': [self.tokenizer.decode(inputs) for inputs in batch['input_ids'].cpu().detach().tolist()]
        }
        return res

    def parse_predictions(self, prediction):
        prediction['topk_tokens'] = [self.tokenizer.convert_ids_to_tokens(item) for item in prediction['topk_tokens']]
        padding_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        while prediction['inputs'] and prediction['inputs'][-1] == padding_id:
            prediction['inputs'].pop()
        prediction['inputs'] = self.tokenizer.decode(prediction['inputs']).strip()
        return prediction


@register_evaluator('classification_demo', BERT_MODELS)
class BERTEvaluatorForClassification(BaseEvaluator):

    def build_model(self):
        return AutoModelForPreTraining.from_pretrained(self.config.arch).to(self.device)

    def preprocessing(self, dataset, **kwargs):
        kwargs['candidate_idx'] = kwargs.get('candidate_idx').to(self.device)
        kwargs['candidate_idx_mask'] = kwargs.get('candidate_idx_mask').to(self.device)
        kwargs['mask_length'] = kwargs['candidate_idx'].shape[-1]
        kwargs['topk'] = getattr(self.config, 'topk', 3)
        candidate_idx = kwargs['candidate_idx'].view(-1, kwargs['mask_length']) # num_labels * num_candidates, maske_length
        candidate_idx_mask = kwargs['candidate_idx_mask'].view(-1, kwargs['mask_length'])

        res = kwargs.get('calibrate_input')
        if res:
            res = collate_fn(res)
            res = {k: v.to(self.device) for k, v in res.items()}
            mask_pos = res.pop('mask_pos')
            with torch.no_grad():
                outputs = self.model(**res)
            logits = get_logits(outputs)
            mask_logits = logits[mask_pos > 0].view(-1, kwargs['mask_length'], logits.shape[-1])
            mask_logits = torch.nn.functional.log_softmax(mask_logits, dim=-1)
            candidate_logits = torch.stack(
                [mask_logits[:, i, :].index_select(-1, candidate_idx[:, i]) for i in range(kwargs['mask_length'])], dim=1)
            kwargs['calibrate_logits'] = torch.sum(torch.mul(candidate_logits, candidate_idx_mask.T.float()), dim=1)/torch.sum(candidate_idx_mask, dim=-1)

        return dataset, kwargs

    def decode(self, batch, **kwargs):
        mask_pos = batch.pop('mask_pos')
        candidate_idx = kwargs.get('candidate_idx')
        candidate_labels = kwargs.get('candidate_labels')
        mask_length = kwargs.get('mask_length')
        candidate_idx_mask = kwargs.get('candidate_idx_mask')
        candidate_num = candidate_idx.shape[1]
        candidate_idx = candidate_idx.view(-1, mask_length) # num_labels * num_candidates, maske_length
        candidate_idx_mask = candidate_idx_mask.view(-1, mask_length)
        calibrate_logits = kwargs.get('calibrate_logits')

        topk = kwargs.get('topk')
        with torch.no_grad():
            outputs = self.model(**batch)
        logits = get_logits(outputs)
        mask_logits = logits[mask_pos > 0].view(-1, mask_length, logits.shape[-1])
        mask_logits = torch.nn.functional.log_softmax(mask_logits, dim=-1)
        candidate_logits = torch.stack(
            [mask_logits[:, i, :].index_select(-1, candidate_idx[:, i]) for i in range(mask_length)], dim=1)
        candidate_logits = torch.sum(torch.mul(candidate_logits, candidate_idx_mask.T.float()), dim=1)/torch.sum(candidate_idx_mask, dim=-1)
        if calibrate_logits is not None:
            candidate_logits -= calibrate_logits
        max_tokens, max_indices = torch.topk(candidate_logits, k=topk)
        max_label_idx, max_cand_idx = max_indices // candidate_num, max_indices % candidate_num
        res = {
            'predictions': [candidate_labels[i] for i in torch.mode(max_label_idx, -1)[0].cpu().detach().numpy()],
            'topk_tokens':  candidate_idx[max_indices].cpu().detach().tolist(),
            'inputs': batch['input_ids'].cpu().detach().tolist()
        }
        return res

    def parse_predictions(self, prediction):
        padding_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        prediction['topk_tokens'] = self.tokenizer.batch_decode(prediction['topk_tokens'], skip_special_tokens=True)
        while prediction['inputs'] and prediction['inputs'][-1] == padding_id:
            prediction['inputs'].pop()
        prediction['inputs'] = self.tokenizer.decode(prediction['inputs']).strip()
        return prediction