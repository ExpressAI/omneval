from sklearn.metrics import accuracy_score

METRICS_REGISTRY = {}

def build_metrics(name):
    return METRICS_REGISTRY[name]


def register_metrics(name):
    """
    New metrics can be added with the :func:`register_metrics`
    """
    def register_metrics_fn(fn):
        if name in METRICS_REGISTRY:
            raise ValueError('Cannot register duplicate metrics ({})'.format(name))
        if not callable(name):
            raise ValueError('metrics must be callable ({})'.format(name))
        METRICS_REGISTRY[name] = fn
        return fn

    return register_metrics_fn


@register_metrics('accuracy')
def accuracy(labels, predictions):
    return accuracy_score(labels, predictions)