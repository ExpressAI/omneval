import torch
import warnings
from omneval.tasks import BaseProcessor
from omneval.utils import difference, pad_input_ids, normalize_raw_text_to_inputs, truncate_text, append_templates_to_inputs
from omneval.registry import register_processor
from .. import BaseProcessor
import logging
import pdb
from transformers import GPT2Tokenizer, GPT2TokenizerFast
import string

warnings.filterwarnings('ignore')


@register_processor('generation')
class ProcessorForGeneration(BaseProcessor):

    def __init__(self, config):
        assert hasattr(config, 'templates'), "prompt templates should be specified."
        super(ProcessorForGeneration, self).__init__(config)
        self.remove_punc = getattr(self.config, "remove_punc", False)

    @property
    def prompt_count(self):
        return len(self.config.templates)

    def prompt_schema(self, pid):
        return self.config.templates[pid]

    def _prompting(self, example, schema, max_length):
        text = ''
        text_length_cnt = 0
        for idx, item in enumerate(schema.split('|')):
            item = item.strip()
            if item in example and isinstance(example[item], str):
                appended_text = normalize_raw_text_to_inputs(example[item], self.remove_punc)
                appended_text, appended_length = truncate_text(appended_text, self.tokenizer,
                                                               max_length - text_length_cnt)
                text_length_cnt += appended_length
                text += appended_text
            # for prompting templates
            else:
                text = append_templates_to_inputs(text, item)
        return text

    def prompting(self, example, prompt_schema, max_length=512):
        assert len(prompt_schema.split('||')) == 2
        encoder_schema, decoder_schema = prompt_schema.split('||')
        encoder_text = self._prompting(example, encoder_schema, max_length)
        res = self.tokenizer(encoder_text.strip())
        text_len = len(res['input_ids'])
        res = pad_input_ids(res, self.max_seq_length, self.padding_id)
        return res


