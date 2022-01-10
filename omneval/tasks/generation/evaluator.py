import torch
from omneval.tasks import BaseEvaluator
from omneval.utils import get_logits, adjust_length
from omneval.registry import register_evaluator
from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from torch.nn.functional import cross_entropy
import pdb
import collections
from omneval.utils import BERT_MODELS, GPT_MODELS, BART_MODELS, T5_MODELS


@register_evaluator('generation', BART_MODELS+T5_MODELS)
class BARTEvaluatorForClassification(BaseEvaluator):

    def build_model(self):
        return AutoModelForPreTraining.from_pretrained(self.config.arch).to(self.device)

    def preprocessing(self, dataset, **kwargs):
        kwargs['topk'] = getattr(self.config, 'topk', 3)
        return dataset, kwargs

    def decode(self, batch, **kwargs):

        topk = kwargs.get('topk')
        with torch.no_grad():
            outputs = self.model.generate(**batch, num_beams=self.config.num_beams, early_stopping=True, max_length=self.config.decode_max_length)
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions = {
            'predictions': pred,
        }
        return predictions

    def parse_predictions(self, prediction):
        return prediction

    def analysis(self, res_list):
        return {}


@register_evaluator('generation', GPT_MODELS)
class GPTEvaluatorForClassification(BaseEvaluator):

    def build_model(self):
        return AutoModelForPreTraining.from_pretrained(self.config.arch).to(self.device)

    def preprocessing(self, dataset, **kwargs):
        kwargs['topk'] = getattr(self.config, 'topk', 3)
        return dataset, kwargs

    def decode(self, batch, **kwargs):

        with torch.no_grad():

            outputs = self.model.generate(**batch, num_beams=self.config.num_beams,
                                          early_stopping=True,
                                          max_length=adjust_length(self.config)+self.config.decode_max_length)
        # encoder_input_sum = batch['attention_mask'].sum(dim=1)
        # new_outputs = torch.ones_like(outputs).to(self.device) * self.padding_id
        # for i in range(outputs.shape[0]):
        #     new_outputs[i][: -encoder_input_sum[i]] = outputs[i][encoder_input_sum[i]: ]
        preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        source = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        new_pred = []
        for i in range(len(preds)):
            new_pred.append(preds[i][len(source[i]): ])
        predictions = {
            'predictions': new_pred,
        }
        return predictions

    def parse_predictions(self, prediction):
        return prediction

    def analysis(self, res_list):
        return {}