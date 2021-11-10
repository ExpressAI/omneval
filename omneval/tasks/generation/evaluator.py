import torch
from omneval.tasks import BaseEvaluator
from omneval.utils import get_logits
from omneval.registry import register_evaluator
from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from torch.nn.functional import cross_entropy
import pdb
import collections


BART_MODELS = ['facebook/bart-base', 'google/bert_for_seq_generation_L-24_bbc_encoder', 'facebook/bart-large',
               't5-base', 'openai-gpt', 'gpt2']



@register_evaluator('generation', BART_MODELS)
class BARTEvaluatorForClassification(BaseEvaluator):

    def build_model(self):
        return AutoModelForPreTraining.from_pretrained(self.config.arch).to(self.device)

    def preprocessing(self, dataset, **kwargs):
        kwargs['topk'] = getattr(self.config, 'topk', 3)
        return dataset, kwargs

    def decode(self, batch, **kwargs):

        topk = kwargs.get('topk')
        with torch.no_grad():
            outputs = self.model.generate(**batch, num_beams=1, early_stopping=True, max_length=self.config.decode_max_length)
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions = {
            'predictions': pred,
        }
        return predictions

    def parse_predictions(self, prediction):
        return prediction

    def analysis(self, res_list):
        return {}