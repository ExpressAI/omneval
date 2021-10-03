import numpy as np
import torch
from torch.utils.data import DataLoader
from .. import BaseEvaluator, register_evaluator
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForPreTraining, AutoModelForCausalLM
from sklearn.metrics import accuracy_score
import pdb
from torch.nn.functional import cross_entropy
import logging
BERT_MODELS = ['bert-base-uncased', 'roberta-base', 'bert-large-uncased', 'roberta-large', 'distilroberta-base',
               'distilbert-base-uncased']
GPT_MODELS = ['openai-gpt', 'gpt2', 'facebook/bart-base', 'google/bert_for_seq_generation_L-24_bbc_encoder']


def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.LongTensor([bz[k] for bz in batch]) for k in keys}


@register_evaluator('classification', BERT_MODELS)
class BERTEvaluatorForClassification(BaseEvaluator):

    def build_model(self, arch):
        return AutoModelForPreTraining.from_pretrained(arch).to(self.device)

    def eval(self, dataset, **kwargs):
        candidate_idx = kwargs.get('candidate_idx').to(self.device)
        candidate_labels = kwargs.get('candidate_labels')
        mask_length = candidate_idx.shape[-1]
        test_dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
        self.model.eval()
        predictions = []
        labels = []

        def decode(outputs, mask_pos, candidate_idx, mask_length):
            """Calculate the predicted probability of answers in the masked position"""
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif hasattr(outputs, 'prediction_logits'):
                logits = outputs.prediction_logits
            else:
                raise NotImplementedError
            mask_logits = logits[mask_pos > 0].view(-1, mask_length, logits.shape[-1])
            candidate_logits = torch.stack(
                [mask_logits[:, i, :].index_select(-1, candidate_idx[:, i]) for i in range(mask_length)], axis=-1)
            candidate_logits = torch.nn.functional.log_softmax(candidate_logits, dim=1)
            return torch.sum(candidate_logits, axis=-1)

        for batch in test_dataloader:
            label = batch.pop('label').cpu().detach().tolist()
            batch = {k: v.to(self.device) for k, v in batch.items()}
            mask_pos = batch.pop('mask_pos')
            with torch.no_grad():
                outputs = self.model(**batch)
            logits = decode(outputs, mask_pos, candidate_idx, mask_length)
            predictions += [candidate_labels[i] for i in logits.argmax(-1).cpu().detach().numpy()]
            labels += label
        return self.metrics_fn(labels, predictions)


@register_evaluator('classification', GPT_MODELS)
class GPTEvaluatorForClassification(BaseEvaluator):

    def build_model(self, arch):
        return AutoModelForCausalLM.from_pretrained(arch).to(self.device)

    def eval(self, dataset, **kwargs):
        candidate_idx = kwargs.get('candidate_idx').to(self.device)
        candidate_labels = kwargs.get('candidate_labels')
        test_dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
        self.model.eval()
        predictions = []
        labels = []
        padding_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        def decode(outputs, tgt_ids):
            """Calculate the PPL of given sentences"""
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif hasattr(outputs, 'prediction_logits'):
                logits = outputs.prediction_logits
            else:
                raise NotImplementedError
            # logits->[seq_length, vocab_size, batch_size], ids ->[seq_length, batch_size]
            shifted_logits = logits[..., :-1, :]
            shifted_tgt_ids = tgt_ids[..., 1: ]
            # TODO：Check details of this cross entropy later
            loss = cross_entropy(shifted_logits.permute(1,2,0), shifted_tgt_ids.permute(1, 0), reduce=False, ignore_index=-100)
            return torch.sum(loss, axis=0) / torch.sum(~shifted_tgt_ids.eq(-100), axis=-1)

        for batch in test_dataloader:
            label = batch.pop('label').cpu().detach().tolist()
            batch = {k: v.to(self.device) for k, v in batch.items()}
            mask_pos = batch.pop('mask_pos')
            ppls = []
            with torch.no_grad():
                for i in range(len(candidate_labels)):
                    # replace masked token with the answer word
                    batch['input_ids'][mask_pos > 0] = candidate_idx[i].repeat(batch['input_ids'].shape[0])
                    tgt_ids = batch['input_ids'].clone()
                    tgt_ids[tgt_ids==padding_id] = -100
                    outputs = self.model(**batch, labels=tgt_ids)
                    ppls.append(decode(outputs, tgt_ids))
            ppls = torch.stack(ppls).transpose(1,0)
            predictions += [candidate_labels[i] for i in ppls.argmin(-1).cpu().detach().numpy()]
            labels += label
        return self.metrics_fn(labels, predictions)



