from omneval.registry import register_metrics
from typing import List
from eaas import Config
from eaas import Client


class BaseMetrics:
    def __init__(self):
        pass
    def compute(self, predictions, references):
        raise NotImplementedError


class EaasMetrics(BaseMetrics):
    def __init__(self, metrics, config=Config(), client=Client()):
        super(EaasMetrics).__init__()
        self.config = config
        self.client = client
        self.metrics = metrics
        self.client.load_config(self.config)

    def compute(self, predictions, references, source=None):
        if source is None:
            inputs = [{'source': '', 'references': ref if isinstance(ref, list) else [ref], 'hypothesis': pred}
                      for pred, ref in zip(predictions, references)]
        else:
            inputs = [{'source':src, 'references': ref if isinstance(ref, list) else [ref], 'hypothesis': pred}
                    for pred, ref, src in zip(predictions, references, source)]
        return self.client.score(inputs, task="sum", metrics=self.metrics, lang='en')


@register_metrics('rouge1')
class Rouge1(EaasMetrics):
    def __init__(self, metrics=('rouge1',), config=Config(), client=Client()):
        super().__init__(metrics, config, client)


@register_metrics('rouge2')
class Rouge2(EaasMetrics):
    def __init__(self, metrics=('rouge2',), config=Config(), client=Client()):
        super(Rouge2).__init__(metrics, config, client)
