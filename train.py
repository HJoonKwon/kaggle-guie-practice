from audioop import cross
import torch
import numpy as np
import os
from tqdm.auto import tqdm
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from config import Config
from loss import cross_entropy_loss
import gc

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

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()

    dataset_size = 0
    running_loss = 0.0
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data["image"].to(device, dtype=torch.float)
        labels = data["label"].to(device, dtype=torch.long)

        batch_size = images.size(0)
        outputs = model(images, labels)
        loss = cross_entropy_loss(outputs, labels)
        loss = loss / Config["n_accumulate"]

        if (step + 1) % Config["n_accumulate"] == 0:
            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])

    return epoch_loss

@torch.inference_mode()
def eval_one_epoch(model, optimizer, dataloader, device, epoch):
    """ compute loss and prediction score during training"""
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data["image"].to(device, torch.float)
        labels = data["label"].to(device, torch.long)

        batch_size = images.size(0)
        outputs = model(images, labels)
        loss = cross_entropy_loss(outputs, labels)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Valid_loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])

        return epoch_loss


def run():
    pass


if __name__ == "__main__":
    run()
