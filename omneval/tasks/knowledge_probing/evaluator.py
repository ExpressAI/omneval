import torch
from torch.utils.data import DataLoader
from omneval.tasks import BaseEvaluator
from omneval.utils import get_logits, collate_fn
from omneval.registry import register_evaluator
from transformers import AutoModelForPreTraining
import re

BERT_MODELS = ['bert-base-uncased', 'roberta-base', 'bert-large-uncased', 'roberta-large', 'distilroberta-base',
               'distilbert-base-uncased']
GPT_MODELS = ['openai-gpt', 'gpt2']
BART_MODELS = ['facebook/bart-base', 'google/bert_for_seq_generation_L-24_bbc_encoder']




@register_evaluator('knowledge_probing', BERT_MODELS+GPT_MODELS+BART_MODELS)
class BERTEvaluatorForKnowledgeProbing(BaseEvaluator):

    def build_model(self):
        return AutoModelForPreTraining.from_pretrained(self.config.arch).to(self.device)

    def decode(self, batch, **kwargs):
        mask_pos = batch.pop('mask_pos')
        with torch.no_grad():
            outputs = self.model(**batch)
        logits = get_logits(outputs)
        mask_logits = logits[mask_pos > 0]
        pred = [i for i in mask_logits.argmax(-1).cpu().detach().numpy()]
        return {'predictions': pred}

    def parse_predictions(self, prediction):
        return prediction

    def analysis(self, res_list):
        return {}