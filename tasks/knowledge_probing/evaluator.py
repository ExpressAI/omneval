import torch
from torch.utils.data import DataLoader
from .. import BaseEvaluator, register_evaluator
from transformers import AutoModelForPreTraining
import re
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

    def eval(self, dataset, **kwargs):
        label_name = getattr(self.config, 'label_name', 'obj_label')
        test_dataloader = DataLoader(dataset, batch_size=8, collate_fn=lambda x: collate_fn(x, exclude=[label_name]))
        self.model.eval()
        predictions = []
        labels = []

        def decode(outputs, mask_pos):
            """Output Logits"""
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif hasattr(outputs, 'prediction_logits'):
                logits = outputs.prediction_logits
            else:
                raise NotImplementedError
            masked_logits = logits[mask_pos > 0]
            return masked_logits

        for batch in tqdm(test_dataloader):
            label = batch.pop(label_name)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            mask_pos = batch.pop('mask_pos')
            with torch.no_grad():
                outputs = self.model(**batch)
            logits = decode(outputs, mask_pos)
            predictions += [normalize(self.tokenizer.decode([i])) for i in logits.argmax(-1).cpu().detach().numpy()]
            labels += [normalize(l) for l in label]
        return self.metrics_fn(labels, predictions)

