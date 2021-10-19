import torch
import pdb


def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.LongTensor([bz[k] for bz in batch]) for k in keys}


def get_logits(outputs):
    if hasattr(outputs, 'logits'):
        logits = outputs.logits
    elif hasattr(outputs, 'prediction_logits'):
        logits = outputs.prediction_logits
    else:
        raise NotImplementedError