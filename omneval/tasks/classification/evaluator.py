import torch
from omneval.tasks import BaseEvaluator
from omneval.utils import get_logits, collate_fn, merge_fn
from omneval.registry import register_evaluator
from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from torch.nn.functional import cross_entropy
import collections
import pdb
from omneval.utils import BERT_MODELS, GPT_MODELS, BART_MODELS, T5_MODELS



class BaseEvaluatorForClassification(BaseEvaluator):

    def preprocessing(self, dataset, **kwargs):
        kwargs['candidate_idx'] = kwargs.get('candidate_idx').to(self.device)
        kwargs['candidate_idx_mask'] = kwargs.get('candidate_idx_mask').to(self.device)
        kwargs['mask_length'] = kwargs['candidate_idx'].shape[-1]
        num_answers_per_label =  kwargs['candidate_idx'].shape[1]
        num_labels = kwargs['candidate_idx'].shape[0]
        kwargs['topk'] = getattr(self.config, 'topk', num_labels * (num_answers_per_label//3)+1)
        calibrate_input = kwargs.get('calibrate_input')
        # TODO: The Calibrate option not well defined for GPT models
        if calibrate_input:
            kwargs['calibrate_logits'] = self.caculate_calibrate_logits(**kwargs)
        return dataset, kwargs

    def caculate_calibrate_logits(self, **kwargs):
        candidate_idx = kwargs['candidate_idx'].view(-1, kwargs['mask_length']) # num_labels * num_candidates, maske_length
        candidate_idx_mask = kwargs['candidate_idx_mask'].view(-1, kwargs['mask_length'])
        calibrate_input = kwargs.get('calibrate_input')
        calibrate_input = collate_fn(calibrate_input)
        calibrate_input = {k: v.to(self.device) for k, v in calibrate_input.items()}
        mask_pos = calibrate_input.pop('mask_pos')
        with torch.no_grad():
            outputs = self.model(**calibrate_input)
        logits = get_logits(outputs)
        mask_logits = logits[mask_pos > 0].view(-1, kwargs['mask_length'], logits.shape[-1])
        mask_logits = torch.nn.functional.log_softmax(mask_logits, dim=-1)
        candidate_logits = torch.stack(
            [mask_logits[:, i, :].index_select(-1, candidate_idx[:, i]) for i in range(kwargs['mask_length'])], dim=1)
        return torch.sum(torch.mul(candidate_logits, candidate_idx_mask.T.float()), dim=1) \
               / torch.sum(candidate_idx_mask, dim=-1)

    def parse_predictions(self, prediction):
        """For classification, return predicted labels and prompted inputs"""
        padding_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        prediction['topk_tokens'] = self.tokenizer.batch_decode(prediction['topk_tokens'], skip_special_tokens=True)
        # TODO: make this as another function
        while prediction['inputs'] and prediction['inputs'][-1] == padding_id:
            prediction['inputs'].pop()
        prediction['inputs'] = self.tokenizer.decode(prediction['inputs']).strip()
        return prediction

    def analysis(self, predictions):
        """For classification, return top5 choices """
        memok = collections.Counter()
        for item in predictions:
            memok.update(item['topk_tokens'])
        return {'top5 choices': [item[0] for item in memok.most_common(5)],
                'calibrated': self.config.calibrate}


@register_evaluator('classification', GPT_MODELS)
class GPTEvaluatorForClassification(BaseEvaluatorForClassification):

    def build_model(self):
        return AutoModelForCausalLM.from_pretrained(self.config.arch).to(self.device)

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
                try:
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
                except:
                    pdb.set_trace()
            candidate_ppls = torch.stack(candidate_ppls).transpose(1, 0)
        max_tokens, max_indices = torch.topk(candidate_ppls, k=topk, largest=False) # find tokens that achieves minimal PPLs
        max_label_idx, max_cand_idx = max_indices // candidate_num, max_indices % candidate_num
        prediction = {
            'predictions': [candidate_labels[i] for i in torch.mode(max_label_idx, -1)[0].cpu().detach().numpy()],
            'topk_tokens':  candidate_idx[max_indices].cpu().detach().tolist(),
            'inputs': batch['input_ids'].cpu().detach().tolist()
        }
        return prediction


@register_evaluator('classification', BERT_MODELS)
class BERTEvaluatorForClassification(BaseEvaluatorForClassification):

    def build_model(self):
        return AutoModelForPreTraining.from_pretrained(self.config.arch).to(self.device)

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
        predictions = {
            'predictions': [candidate_labels[i] for i in torch.mode(max_label_idx, -1)[0].cpu().detach().numpy()],
            'topk_tokens':  candidate_idx[max_indices].cpu().detach().tolist(),
            'inputs': batch['input_ids'].cpu().detach().tolist()
        }
        return predictions


@register_evaluator('classification', BART_MODELS)
class BARTEvaluatorForClassification(BaseEvaluatorForClassification):

    def build_model(self):
        return AutoModelForSeq2SeqLM.from_pretrained(self.config.arch).to(self.device)

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
        predictions = {
            'predictions': [candidate_labels[i] for i in torch.mode(max_label_idx, -1)[0].cpu().detach().numpy()],
            'topk_tokens':  candidate_idx[max_indices].cpu().detach().tolist(),
            'inputs': batch['input_ids'].cpu().detach().tolist()
        }
        return predictions


@register_evaluator('classification', T5_MODELS)
class T5EvaluatorForClassification(BaseEvaluatorForClassification):

    def build_model(self):
        return AutoModelForSeq2SeqLM.from_pretrained(self.config.arch).to(self.device)

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
        batch['labels'] = torch.ones((batch['input_ids'].shape[0], 1+mask_length), dtype=torch.long).to(self.device) * self.mask_token_id

        topk = kwargs.get('topk')
        with torch.no_grad():
            outputs = self.model(**batch)
        logits = get_logits(outputs)
        mask_logits = logits[:, 1: , :].view(-1, mask_length, logits.shape[-1])
        mask_logits = torch.nn.functional.log_softmax(mask_logits, dim=-1)
        candidate_logits = torch.stack(
            [mask_logits[:, i, :].index_select(-1, candidate_idx[:, i]) for i in range(mask_length)], dim=1)
        candidate_logits = torch.sum(torch.mul(candidate_logits, candidate_idx_mask.T.float()), dim=1)/torch.sum(candidate_idx_mask, dim=-1)
        if calibrate_logits is not None:
            candidate_logits -= calibrate_logits
        max_tokens, max_indices = torch.topk(candidate_logits, k=topk)
        max_label_idx, max_cand_idx = max_indices // candidate_num, max_indices % candidate_num
        predictions = {
            'predictions': [candidate_labels[i] for i in torch.mode(max_label_idx, -1)[0].cpu().detach().numpy()],
            'topk_tokens':  candidate_idx[max_indices].cpu().detach().tolist(),
            'inputs': batch['input_ids'].cpu().detach().tolist()
        }
        return predictions

    def caculate_calibrate_logits(self, **kwargs):
        candidate_idx = kwargs['candidate_idx'].view(-1, kwargs['mask_length']) # num_labels * num_candidates, maske_length
        candidate_idx_mask = kwargs['candidate_idx_mask'].view(-1, kwargs['mask_length'])
        calibrate_input = kwargs.get('calibrate_input')
        calibrate_input = collate_fn(calibrate_input)
        calibrate_input = {k: v.to(self.device) for k, v in calibrate_input.items()}
        mask_pos = calibrate_input.pop('mask_pos')
        mask_length = kwargs.get('mask_length')
        label = calibrate_input.pop(self.label_name)
        calibrate_input['labels'] = torch.ones((calibrate_input['input_ids'].shape[0], 1+mask_length), dtype=torch.long).to(self.device) * self.mask_token_id
        with torch.no_grad():
            outputs = self.model(**calibrate_input)
        logits = get_logits(outputs)
        mask_logits = logits[:, 1: , :].view(-1, mask_length, logits.shape[-1])
        mask_logits = torch.nn.functional.log_softmax(mask_logits, dim=-1)
        candidate_logits = torch.stack(
            [mask_logits[:, i, :].index_select(-1, candidate_idx[:, i]) for i in range(kwargs['mask_length'])], dim=1)
        kwargs['calibrate_logits'] = torch.sum(torch.mul(candidate_logits, candidate_idx_mask.T.float()),
                                               dim=1) / torch.sum(candidate_idx_mask, dim=-1)

