import torch
from torch.utils.data import DataLoader
from .. import BaseEvaluator, register_evaluator, get_logits
from transformers import AutoModelForPreTraining
import re
import pdb
from tqdm import tqdm
BERT_MODELS = ['bert-base-uncased', 'roberta-base', 'bert-large-uncased', 'roberta-large', 'distilroberta-base',
               'distilbert-base-uncased']
GPT_MODELS = ['openai-gpt', 'gpt2']
BART_MODELS = ['facebook/bart-base', 'google/bert_for_seq_generation_L-24_bbc_encoder']


def collate_fn(batch, exclude=[]):
    keys = batch[0].keys()
    return {k: (torch.LongTensor([bz[k] for bz in batch]) if k not in exclude else [bz[k] for bz in batch]) for k in keys}


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    return text


@register_evaluator('knowledge_probing', BERT_MODELS)
class BERTEvaluatorForKnowledgeProbing(BaseEvaluator):

    def build_model(self, arch):
        return AutoModelForPreTraining.from_pretrained(arch).to(self.device)

    def decode(self, batch, **kwargs):
        mask_pos = batch.pop('mask_pos')
        with torch.no_grad():
            outputs = self.model(**batch)
        logits = get_logits(outputs)
        mask_logits = logits[mask_pos > 0]
        return [i for i in mask_logits.argmax(-1).cpu().detach().numpy()]