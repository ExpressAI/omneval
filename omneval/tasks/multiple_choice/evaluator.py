import torch
from omneval.tasks import BaseEvaluator
from omneval.utils import get_logits, collate_fn, merge_fn
from omneval.registry import register_evaluator
from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from torch.nn.functional import cross_entropy
import pdb
from omneval.utils import BERT_MODELS, GPT_MODELS, BART_MODELS, T5_MODELS
import collections


class BaseEvaluatorForMultipleChoice(BaseEvaluator):
    def parse_predictions(self, prediction):
        padding_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        while prediction['inputs'] and prediction['inputs'][-1] == padding_id:
            prediction['inputs'].pop()
        prediction['inputs'] = self.tokenizer.decode(prediction['inputs']).strip()
        return prediction
    @property
    def exclude_collate_features(self):
        return [self.label_name, 'labels_ids', 'labels_masks']


@register_evaluator('multiple_choice', GPT_MODELS)
class GPTEvaluatorForMultipleChoice(BaseEvaluatorForMultipleChoice):

    def build_model(self):
        return AutoModelForCausalLM.from_pretrained(self.config.arch).to(self.device)

    def decode(self, batch, **kwargs):
        candidate_idx = batch.pop('labels_ids')
        candidate_labels = kwargs.get('candidate_labels')
        candidate_idx_mask = batch.pop('labels_masks')
        mask_length = batch.pop('mask_length')
        mask_pos = batch.pop('mask_pos')
        candidate_num = len(candidate_labels)
        candidate_ppls = []
        # Iterate over all possible candidates and calculate ppls
        input_ids = []
        old_input_ids = batch['input_ids'].clone()
        with torch.no_grad():
            for idx in range(batch['input_ids'].shape[0]):
                masked = mask_pos[idx]
                cand = torch.tensor(candidate_idx[idx]).to(self.device)
                cand_mask = torch.tensor(candidate_idx_mask[idx]).to(self.device)
                input_id = batch['input_ids'][idx]
                input_ids_one_example = []
                for i in range(candidate_num):
                    try:
                        input_id[masked > 0] = cand[i]
                    except:
                        pdb.set_trace()
                    tgt_ids = input_id.clone()
                    input_ids_one_example.append(tgt_ids)
                input_ids_one_example = torch.stack(input_ids_one_example)
                input_ids.append(input_ids_one_example)
            input_ids = torch.stack(input_ids)
            for i in range(candidate_num):
                batch['input_ids'] = input_ids[:, i, :].contiguous()
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

        predictions = {
            'predictions': [candidate_labels[i] for i in torch.argmin(candidate_ppls, dim=-1).cpu().detach().numpy()],
            'inputs': old_input_ids.cpu().detach().tolist()
        }
        return predictions


@register_evaluator('multiple_choice', BERT_MODELS)
class BERTEvaluatorForMultipleChoice(BaseEvaluatorForMultipleChoice):

    def build_model(self):
        return AutoModelForPreTraining.from_pretrained(self.config.arch).to(self.device)

    def decode(self, batch, **kwargs):
        mask_pos = batch.pop('mask_pos')
        candidate_idx = batch.pop('labels_ids')
        candidate_idx_mask = batch.pop('labels_masks')
        candidate_labels = kwargs.get('candidate_labels')
        mask_length = batch.pop('mask_length')
        with torch.no_grad():
            outputs = self.model(**batch)
        logits = get_logits(outputs)
        candidate_logits = []
        for idx, logit in enumerate(logits):
            masked = mask_pos[idx]
            cand = torch.tensor(candidate_idx[idx]).to(self.device)
            cand_mask = torch.tensor(candidate_idx_mask[idx]).to(self.device)
            mask_logit = logit[masked > 0].view(mask_length[idx], logit.shape[-1])
            mask_logit = torch.nn.functional.log_softmax(mask_logit, dim=-1)
            candidate_logit = torch.stack(
                [mask_logit[i, :].index_select(-1, cand[:, i]) for i in range(mask_length[idx])], dim=1)
            candidate_logit = torch.sum(torch.mul(candidate_logit, cand_mask.float()), dim=1) / torch.sum(
                cand_mask, dim=-1)
            candidate_logits.append(candidate_logit)
        candidate_logits = torch.stack(candidate_logits)
        predictions = {
            'predictions': [candidate_labels[i] for i in torch.argmax(candidate_logits, dim=-1).cpu().detach().numpy()],
            'inputs': batch['input_ids'].cpu().detach().tolist()
        }
        return predictions


@register_evaluator('multiple_choice', BART_MODELS)
class BARTEvaluatorForMultipleChoice(BERTEvaluatorForMultipleChoice):

    def build_model(self):
        return AutoModelForSeq2SeqLM.from_pretrained(self.config.arch).to(self.device)



@register_evaluator('multiple_choice', T5_MODELS)
class T5EvaluatorFor(BaseEvaluatorForMultipleChoice):

    def build_model(self):
        return AutoModelForSeq2SeqLM.from_pretrained(self.config.arch).to(self.device)

    def decode(self, batch, **kwargs):
        mask_pos = batch.pop('mask_pos')
        candidate_idx = batch.pop('labels_ids')
        candidate_idx_mask = batch.pop('labels_masks')
        candidate_labels = kwargs.get('candidate_labels')
        mask_length = batch.pop('mask_length')
        with torch.no_grad():
            outputs = self.model(**batch)
        logits = get_logits(outputs)
        candidate_logits = []
        for idx, logit in enumerate(logits):
            masked = mask_pos[idx]
            cand = torch.tensor(candidate_idx[idx]).to(self.device)
            cand_mask = torch.tensor(candidate_idx_mask[idx]).to(self.device)
            mask_logit = logit[masked > 0].view(mask_length[idx], logit.shape[-1])
            mask_logit = torch.nn.functional.log_softmax(mask_logit, dim=-1)
            candidate_logit = torch.stack(
                [mask_logit[i, :].index_select(-1, cand[:, i]) for i in range(mask_length[idx])], dim=1)
            candidate_logit = torch.sum(torch.mul(candidate_logit, cand_mask.float()), dim=1) / torch.sum(
                cand_mask, dim=-1)
            candidate_logits.append(candidate_logit)
        candidate_logits = torch.stack(candidate_logits)
        predictions = {
            'predictions': [candidate_labels[i] for i in torch.argmax(candidate_logits, dim=-1).cpu().detach().numpy()],
            'inputs': batch['input_ids'].cpu().detach().tolist()
        }
        return predictions
    def decode(self, batch, **kwargs):
        mask_pos = batch.pop('mask_pos')
        candidate_idx = batch.pop('labels_ids')
        candidate_idx_mask = batch.pop('labels_masks')
        candidate_labels = kwargs.get('candidate_labels')
        mask_length = batch.pop('mask_length')
        candidate_num = len(candidate_labels)
        batch['labels'] = torch.ones((batch['input_ids'].shape[0], 1 + mask_length), dtype=torch.long).to(
            self.device) * self.mask_token_id

        topk = kwargs.get('topk')
        with torch.no_grad():
            outputs = self.model(**batch)
        logits = get_logits(outputs)
        mask_logits = logits[:, 1:, :].view(-1, mask_length, logits.shape[-1])
        mask_logits = torch.nn.functional.log_softmax(mask_logits, dim=-1)
        candidate_logits = torch.stack(
            [mask_logits[:, i, :].index_select(-1, candidate_idx[:, i]) for i in range(mask_length)], dim=1)
        candidate_logits = torch.sum(torch.mul(candidate_logits, candidate_idx_mask.T.float()), dim=1) / torch.sum(
            candidate_idx_mask, dim=-1)
        if calibrate_logits is not None:
            candidate_logits -= calibrate_logits
        max_tokens, max_indices = torch.topk(candidate_logits, k=topk)
        max_label_idx, max_cand_idx = max_indices // candidate_num, max_indices % candidate_num
        predictions = {
            'predictions': [candidate_labels[i] for i in torch.mode(max_label_idx, -1)[0].cpu().detach().numpy()],
            'topk_tokens': candidate_idx[max_indices].cpu().detach().tolist(),
            'inputs': batch['input_ids'].cpu().detach().tolist()
        }
        return predictions

    def caculate_calibrate_logits(self, **kwargs):
        candidate_idx = kwargs['candidate_idx'].view(-1,
                                                     kwargs['mask_length'])  # num_labels * num_candidates, maske_length
        candidate_idx_mask = kwargs['candidate_idx_mask'].view(-1, kwargs['mask_length'])
        calibrate_input = kwargs.get('calibrate_input')
        calibrate_input = collate_fn(calibrate_input)
        calibrate_input = {k: v.to(self.device) for k, v in calibrate_input.items()}
        mask_pos = calibrate_input.pop('mask_pos')
        mask_length = kwargs.get('mask_length')
        label = calibrate_input.pop('label')
        calibrate_input['labels'] = torch.ones((calibrate_input['input_ids'].shape[0], 1 + mask_length),
                                               dtype=torch.long).to(self.device) * self.mask_token_id
        with torch.no_grad():
            outputs = self.model(**calibrate_input)
        logits = get_logits(outputs)
        mask_logits = logits[:, 1:, :].view(-1, mask_length, logits.shape[-1])
        mask_logits = torch.nn.functional.log_softmax(mask_logits, dim=-1)
        candidate_logits = torch.stack(
            [mask_logits[:, i, :].index_select(-1, candidate_idx[:, i]) for i in range(kwargs['mask_length'])], dim=1)
        kwargs['calibrate_logits'] = torch.sum(torch.mul(candidate_logits, candidate_idx_mask.T.float()),
                                               dim=1) / torch.sum(candidate_idx_mask, dim=-1)


