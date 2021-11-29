import torch
from omneval.tasks import BaseEvaluator
from omneval.utils import get_logits, collate_fn, check_if_bpe_tokenizer
from omneval.registry import register_evaluator
from transformers import AutoModelForPreTraining
import os
import pdb
from omneval.utils import BERT_MODELS, GPT_MODELS, BART_MODELS, T5_MODELS




class BaseEvaluatorForKnowledgeProbing(BaseEvaluator):

    def __init__(self, config):
        super().__init__(config)
        self.vocab_file = getattr(self.config, "vocab_file", None)
        if self.vocab_file:
            self.vocab_id_list = torch.tensor(self.get_common_vocab_id_list(), dtype=torch.long, device=self.device)
        else:
            self.vocab_id_list = None
            self.vocab_idx = None

    def get_common_vocab_id_list(self):
        assert os.path.exists(self.vocab_file), "vocab file not exist"
        common_vocabs = set()
        with open(self.vocab_file, 'r') as f:
            vocab_id_list = f.readlines()
        for vocab in vocab_id_list:
            vocab = vocab.strip()
            if check_if_bpe_tokenizer(self.tokenizer):
                vocab = ' '+vocab
                aux = self.tokenizer.tokenize(vocab)
                if len(aux) > 1:
                    continue
                vocab = aux[0]
            vocab_id = self.tokenizer.convert_tokens_to_ids(vocab)
            if vocab_id not in (self.mask_token_id, self.padding_id, self.tokenizer.unk_token_id):
                common_vocabs.add(vocab_id)
        return list(common_vocabs)

    def parse_predictions(self, prediction):
        pred = self.tokenizer.convert_ids_to_tokens(int(prediction['predictions']))
        prediction['predict_label'] = pred
        return prediction

    def analysis(self, res_list):
        return {}


@register_evaluator('knowledge_probing', BERT_MODELS+BART_MODELS)
class BERTEvaluatorForKnowledgeProbing(BaseEvaluatorForKnowledgeProbing):

    def build_model(self):
        return AutoModelForPreTraining.from_pretrained(self.config.arch).to(self.device)

    def decode(self, batch, **kwargs):
        mask_pos = batch.pop('mask_pos')
        with torch.no_grad():
            outputs = self.model(**batch)
        logits = get_logits(outputs)
        # mask_logits = logits[mask_pos > 0]
        # pred = [i for i in mask_logits.argmax(-1).cpu().detach().numpy()]
        mask_logits = logits[mask_pos > 0].index_select(dim=-1, index=self.vocab_id_list)
        pred = [self.vocab_id_list[i].item() for i in mask_logits.argmax(-1).cpu().detach().numpy()]
        return {'predictions': pred}



@register_evaluator('knowledge_probing', GPT_MODELS)
class GPTEvaluatorForKnowledgeProbing(BaseEvaluatorForKnowledgeProbing):

    def build_model(self):
        return AutoModelForPreTraining.from_pretrained(self.config.arch).to(self.device)

    def decode(self, batch, **kwargs):
        mask_pos = batch.pop('mask_pos')
        # TODO: whether we need to shift the logits
        mask_pos = mask_pos[:, 1:]
        batch['input_ids'][batch['input_ids'] == self.mask_token_id] \
            = torch.tensor(self.padding_id).repeat(batch['input_ids'].shape[0]).to(self.device)
        with torch.no_grad():
            outputs = self.model(**batch)
        logits = get_logits(outputs)
        mask_logits = logits[:, :-1, :][mask_pos > 0].index_select(dim=-1, index=self.vocab_id_list)
        pred = [self.vocab_id_list[i].item() for i in mask_logits.argmax(-1).cpu().detach().numpy()]
        return {'predictions': pred}
