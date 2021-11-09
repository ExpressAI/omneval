import torch
from omneval.tasks import BaseEvaluator
from omneval.utils import get_logits, collate_fn
from omneval.registry import register_evaluator
from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from torch.nn.functional import cross_entropy
import collections
import pdb

BERT_MODELS = ['bert-base-uncased', 'roberta-base', 'bert-large-uncased', 'roberta-large', 'distilroberta-base',
               'distilbert-base-uncased']
GPT_MODELS = ['openai-gpt', 'gpt2']
BART_MODELS = ['facebook/bart-base', 'google/bert_for_seq_generation_L-24_bbc_encoder', 'facebook/bart-large',
               't5-base']

#
# def collate_fn(batch):
#     if callable(getattr(batch, "keys", None)):
#         keys = batch.keys()
#         return {k: torch.LongTensor([batch[k]]) for k in keys}
#     else:
#         keys = batch[0].keys()
#         return {k: torch.LongTensor([bz[k] for bz in batch]) for k in keys}


@register_evaluator('relation_extraction', BERT_MODELS+BART_MODELS)
class BERTEvaluatorForRelationExtraction(BaseEvaluator):
    def build_model(self):
        return AutoModelForPreTraining.from_pretrained(self.config.arch).to(self.device)

    def preprocessing(self, dataset, **kwargs):
        kwargs['candidate_idx'] = kwargs.get('candidate_idx').to(self.device)
        kwargs['candidate_idx_mask'] = kwargs.get('candidate_idx_mask').to(self.device)
        kwargs['topk'] = getattr(self.config, 'topk', 3)
        candidate_idx = kwargs['candidate_idx'].view(kwargs['candidate_idx'].shape[0], -1)    # num_labels * (mask_length1+2+3)
        candidate_idx_mask = kwargs['candidate_idx_mask'].view(kwargs['candidate_idx'].shape[0], -1)
        # TODO: Not working for now
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
        res = {
            'predictions': [candidate_labels[i] for i in candidate_logits.argmax(dim=1).cpu().detach().numpy()],
            'inputs': batch['input_ids'].cpu().detach().tolist()
        }
        return res

    def parse_predictions(self, prediction):
        padding_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        while prediction['inputs'] and prediction['inputs'][-1] == padding_id:
            prediction['inputs'].pop()
        prediction['inputs'] = self.tokenizer.decode(prediction['inputs']).strip()
        return prediction

    def analysis(self, res_list):
        return {}

@register_evaluator('relation_extraction', GPT_MODELS)
class GPTEvaluatorForRelationExtraction(BaseEvaluator):
    def build_model(self):
        return AutoModelForPreTraining.from_pretrained(self.config.arch).to(self.device)

    def preprocessing(self, dataset, **kwargs):
        kwargs['candidate_idx'] = kwargs.get('candidate_idx').to(self.device)
        kwargs['candidate_idx_mask'] = kwargs.get('candidate_idx_mask').to(self.device)
        kwargs['topk'] = getattr(self.config, 'topk', 3)
        candidate_idx = kwargs['candidate_idx'].view(kwargs['candidate_idx'].shape[0], -1)    # num_labels * (mask_length1+2+3)
        candidate_idx_mask = kwargs['candidate_idx_mask'].view(kwargs['candidate_idx'].shape[0], -1)
        # TODO: Not working for now
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
        candidate_num = candidate_idx.shape[1]
        mask_length = kwargs.get('mask_length')
        candidate_idx = candidate_idx.view(-1, mask_length) # num_labels * num_candidates, maske_length
        calibrate_logits = kwargs.get('calibrate_logits')
        topk = kwargs.get('topk')
        candidate_ppls = []
        # Iterate over all possible candidates and calculate ppls
        with torch.no_grad():
            for i in range(candidate_idx.shape[0]):
                batch['input_ids'][mask_pos > 0] = candidate_idx[i].repeat(batch['input_ids'].shape[0])
                tgt_ids = batch['input_ids'].clone()
                tgt_ids[tgt_ids == self.padding_id] = -100
                outputs = self.model(**batch, labels=tgt_ids)
                logits = get_logits(outputs)
                # logits->[seq_length, vocab_size, batch_size], ids ->[seq_length, batch_size]
                shifted_logits = logits[..., :-1, :]
                shifted_tgt_ids = tgt_ids[..., 1:]
                loss = cross_entropy(shifted_logits.permute(1, 2, 0), shifted_tgt_ids.permute(1, 0), reduce=False,
                                     ignore_index=-100)
                candidate_ppls.append(torch.sum(loss, axis=0) / torch.sum(~shifted_tgt_ids.eq(-100), axis=-1))
            candidate_ppls = torch.stack(candidate_ppls).transpose(1, 0)
        prediction = {
            'predictions': [candidate_labels[i] for i in candidate_ppls.argmax(dim=1).cpu().detach().numpy()],
            'inputs': batch['input_ids'].cpu().detach().tolist()
        }
        return prediction

    def parse_predictions(self, prediction):
        padding_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        while prediction['inputs'] and prediction['inputs'][-1] == padding_id:
            prediction['inputs'].pop()
        prediction['inputs'] = self.tokenizer.decode(prediction['inputs']).strip()
        return prediction

    def analysis(self, res_list):
        return {}