import torch
import numpy as np
import os
from torch.backends import cudnn
from evaluate import eval_one_epoch

def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    cudnn.deterministic = True
    cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_scheduler():
    pass


def get_optimizer():
    pass


def get_dataloaders():
    pass


def train_one_epoch():
    pass


def run():
    pass
