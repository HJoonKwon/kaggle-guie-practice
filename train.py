from audioop import cross
import torch
import numpy as np
import os
from tqdm.auto import tqdm
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import copy
from collections import defaultdict
import pandas as pd
from config import Config
from sklearn.model_selection import StratifiedKFold
from dataset import GUIEDataset, alb_transforms
from loss import cross_entropy_loss
from model import GUIEModel

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
    optimizer = optim.Adam(model.parameters(),
                           lr=Config['learning_rate'],
                           weight_decay=Config['weight_decay'])
    return optimizer


def create_folds(df):
    skf_kwargs = {'X':df, 'y':df['label_id']}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
    df['kfold'] = -1
    for fold_id, (train_idx, valid_idx) in enumerate(skf.split(**skf_kwargs)):
        df.loc[valid_idx, "kfold"] = fold_id

    # classes = sorted(df['label_id'].unique())
    # label_mapping = dict(zip(classes, list(range(len(classes)))))
    return df


def generate_df(data_dir):
    df = pd.read_csv(os.path.join(data_dir, 'train.csv'),
                     low_memory=False,
                     squeeze=True)
    df = create_folds(df)
    return df


def get_scheduler(optimizer):
    if Config['scheduler'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config['T_max'], eta_min=Config['min_lr'])
    elif Config['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=Config['T_0'], eta_min=Config['min_lr'])
    elif Config['scheduler'] == None:
        return None

    return scheduler


def get_dataloaders(df, fold, dataset, alb_transforms):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    train_dataset = dataset(df_train, transforms=alb_transforms["train"])
    valid_dataset = dataset(df_valid, transforms=alb_transforms["valid"])
    train_loader = DataLoader(train_dataset,
                              batch_size=Config['train_batch_size'],
                              num_workers=2,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=Config['valid_batch_size'],
                              num_workers=2,
                              shuffle=False,
                              pin_memory=True)
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

        bar.set_postfix(Epoch=epoch,
                        Train_Loss=epoch_loss,
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

        bar.set_postfix(Epoch=epoch,
                        Valid_loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])

        # Empty CUDA cache
        if device != torch.device('cpu'):
            torch.cuda.empty_cache()

        return epoch_loss


def run(model: torch.nn.Module, optimizer: torch.optim.optimizer,
        scheduler: torch.optim.lr_scheduler, device: torch.device,
        num_epochs: int):
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)

    df = generate_df(Config["data_dir"])
    train_loader, valid_loader = get_dataloaders(df, Config["fold"],
                                                 GUIEDataset, alb_transforms)

    for epoch in range(1, num_epochs + 1):
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler,
                                           train_loader, device, epoch)
        val_epoch_loss = eval_one_epoch(model,
                                        valid_loader,
                                        device=device,
                                        epoch=epoch)
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)

        # deep copy the model
        if val_epoch_loss <= best_epoch_loss:
            print(f"{b_}Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = os.path.join(Config['ckpr_dir'],f"Loss{best_epoch_loss:.4f}_epoch{epoch:.0f}.bin")
            torch.save(model.state_dict(), PATH)
            print(f"Model Saved{sr_}")
        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print(f"Best Loss: {best_epoch_loss:.4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history

if __name__ == "__main__":
    model = GUIEModel(Config["model_name"], embedding_size=64, target_size=[224, 224])
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    device = Config["device"]
    num_epochs = Config["num_epochs"]
    model, history = run(model, optimizer, scheduler, device, num_epochs)
