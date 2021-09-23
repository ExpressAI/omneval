import torch
import pdb


def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.LongTensor([bz[k] for bz in batch]) for k in keys}


def set_config(cls, config):
    for k, v in config.items():
        setattr(cls, k, v)