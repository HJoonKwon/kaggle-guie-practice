import torch
import numpy as np
import os
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from evaluate import eval_one_epoch
from config import Config

##TODO:: implement DDP for multi-gpu training

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


def get_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=Config['learning_rate'],
                       weight_decay=Config['weight_decay'])
    return optimizer


def get_scheduler(optimizer):
    if Config['scheduler'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=Config['T_max'],
                                                   eta_min=Config['min_lr'])
    elif Config['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=Config['T_0'],
                                                             eta_min=Config['min_lr'])
    elif Config['scheduler'] == None:
        return None

    return scheduler


def get_dataloaders(df, fold, dataset, alb_transforms):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    train_dataset = dataset(df_train, transforms=alb_transforms["train"])
    valid_dataset = dataset(df_valid, transforms=alb_transforms["valid"])
    train_loader = DataLoader(train_dataset, batch_size=Config['train_batch_size'],
                              num_workers=2, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=Config['valid_batch_size'],
                              num_workers=2, shuffle=False, pin_memory=True)
    return train_loader, valid_loader

def train_one_epoch():
    pass


def run():
    pass


if __name__ == "__main__":
    run()
