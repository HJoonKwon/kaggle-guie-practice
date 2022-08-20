import torch


@torch.inference_mode()
def eval_one_epoch():
    """ compute loss and prediction score during training"""
    pass

@torch.inference_mode()
def compute_metric():
    """ compute competition metric"""
    pass
