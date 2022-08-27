from audioop import cross
from cProfile import label
import torch
import numpy as np
import os
from tqdm.auto import tqdm
from typing import Optional
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import torch.optim as optim
import torch.multiprocessing as mp
import time
import copy
from collections import defaultdict
import pandas as pd
import visdom
from config import ConfigType
from sklearn.model_selection import StratifiedKFold
from dataset import GUIEDataset, alb_transforms
from loss import cross_entropy_loss
from model import GUIEModel
from preprocess import preprocess_main
from utils import init_for_distributed, setup_for_distributed

##TODO:: freezing model backbone
##TODO:: different model output for training/inference (https://www.kaggle.com/code/motono0223/guie-clip-tensorflow-train-example/notebook)
##TODO:: syncnorm?
##TODO:: best model?
##TODO:: wandb


def set_seed(seed: int = 42) -> None:
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    cudnn.deterministic = True
    cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_optimizer(model: torch.nn.Module, opts: ConfigType) -> optim.Optimizer:
    if opts.optim_name == "SGD":
        optimizer = optim.SGD(params=model.parameters(),
                              lr=opts.learning_rate,
                              weight_decay=opts.weight_decay,
                              momentum=opts.momentum)
    elif opts.optim_name == "Adam":
        optimizer = optim.Adam(params=model.parameters(),
                               lr=opts.learning_rate,
                               weight_decay=opts.weight_decay)
    else:
        print("Not an appropriate name for optimizer")
        raise NameError
    return optimizer


def create_folds(df: pd.DataFrame) -> pd.DataFrame:
    skf_kwargs = {'X': df, 'y': df['label_id']}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)
    df['kfold'] = -1
    for fold_id, (train_idx, valid_idx) in enumerate(skf.split(**skf_kwargs)):
        df.loc[valid_idx, "kfold"] = fold_id

    # classes = sorted(df['label_id'].unique())
    # label_mapping = dict(zip(classes, list(range(len(classes)))))
    return df


def generate_df() -> pd.DataFrame:
    # df = pd.read_csv(os.path.join(data_dir, 'train.csv'),
    #                  low_memory=False,
    #                  squeeze=True)
    df = preprocess_main()
    df = create_folds(df)
    return df


def get_scheduler(
        optimizer: torch.optim.Optimizer,
        opts: ConfigType) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if opts.scheduler == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=opts.T_max,
                                                         eta_min=opts.min_lr)
    elif opts.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=opts.T_0, eta_min=opts.min_lr)
    elif opts.scheduler == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                              step_size=opts.step_size,
                                              gamma=opts.gamma)
    elif opts.scheduler == None:
        print("No scheduler. Constant learning rate will be applied")
        return None

    return scheduler


def get_dataloaders(df: pd.DataFrame, alb_transforms: dict, opts: ConfigType):
    fold = opts.fold
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    train_dataset = GUIEDataset(df_train, transforms=alb_transforms["train"])
    valid_dataset = GUIEDataset(df_valid, transforms=alb_transforms["valid"])

    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    valid_sampler = DistributedSampler(dataset=valid_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=opts.train_sample_per_gpu,
                              sampler=train_sampler,
                              num_workers=opts.num_workers_per_gpu,
                              shuffle=False,
                              pin_memory=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=opts.valid_sample_per_gpu,
                              sampler=valid_sampler,
                              num_workers=opts.num_workers_per_gpu,
                              shuffle=False,
                              pin_memory=True)
    return {
        "train": (train_loader, train_sampler),
        "valid": (valid_loader, valid_sampler)
    }


def train_one_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler._LRScheduler,
                    dataloader: DataLoader, epoch: int) -> float:
    model.train()

    dataset_size = 0
    running_loss = 0.0
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data["image"].type(torch.Tensor.float).cuda()
        labels = data["label"].type(torch.Tensor.long).cuda()

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
def eval_one_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   dataloader: DataLoader, epoch: int):
    """ compute loss and prediction score during training"""
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data["image"].type(torch.Tensor.float).cuda()
        labels = data["label"].type(torch.Tensor.long).cuda()

        batch_size = images.size(0)
        outputs = model(images, labels)
        loss = cross_entropy_loss(outputs, labels)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch,
                        Valid_loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])

        return epoch_loss


def main_worker(rank, df: pd.DataFrame, opts: ConfigType):
    local_gpu_id = init_for_distributed(rank, opts)
    vis = visdom.Visdom(port=opts.port)

    # data loader
    loaders_dict = get_dataloaders(df, alb_transforms, opts)
    train_loader, train_sampler = loaders_dict["train"]
    valid_loader, valid_sampler = loaders_dict["valid"]

    # model
    model = GUIEModel(opts)
    model = model.cuda(local_gpu_id)
    model = DDP(module=model, device_ids=[local_gpu_id])

    # criterion
    criterion = torch.nn.CrossEntropyLoss().to(local_gpu_id)

    # optimizer
    optimizer = get_optimizer(model=model, opts=opts)

    # scheduler
    scheduler = get_scheduler(optimizer=optimizer, opts=opts)

    for epoch in range(opts.start_epoch, opts.epoch):
        tic = time.time()

        # train
        model.train()
        train_sampler.set_epoch(epoch)
        bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, (images, labels) in bar:
            images = images.to(local_gpu_id)
            labels = labels.to(local_gpu_id)
            outputs = model(images, labels)

            # update
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # retreive lr
            lr = optimizer.param_groups[0]['lr']

            # time
            toc = time.time()

            # visualization
            if (step % opts.vis_step == 0
                    or step == len(train_loader) - 1) and opts.rank == 0:

                # bar.set_postfix(Epoch=epoch,
                #                 Iter=step,
                #                 Train_Loss=loss.item(),
                #                 LR=lr,
                #                 Time=toc - tic)

                print(
                    f"Epoch [{epoch}/{opts.epoch}], Iter [{step}/{len(train_loader)-1}], Loss: {loss.item():.4f},\n"
                    f"LR: {lr:.5f}, Time: {toc-tic:.2f}")

                vis.line(X=torch.ones(
                    (1, 1)) * step + epoch * len(train_loader),
                         Y=torch.Tensor([loss]).unsqueeze(0),
                         update='append',
                         win='loss',
                         opts=dict(x_label='step',
                                   y_label='loss',
                                   title='loss',
                                   legend=['total_loss']))

        if opts.rank == 0:
            if not os.path.exists(opts.save_path):
                os.mkdir(opts.save_path)

            if scheduler:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }
            else:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }
            torch.save(
                checkpoint,
                os.path.join(opts.save_path,
                             opts.save_file_name + f'.{epoch}.pth.tar'))
            print(f"save pth.tar {epoch} epoch!")

        # test
        if opts.rank == 0:
            model.eval()

            val_avg_loss = 0
            correct_top1 = 0
            correct_top5 = 0
            total = 0

            with torch.no_grad():
                bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
                for step, (images, labels) in bar:
                    images = images.to(local_gpu_id)
                    labels = labels.to(local_gpu_id)
                    outputs = model(images, labels)
                    loss = criterion(outputs, labels)
                    val_avg_loss += loss.item()

                    # top 1 (compare the best pred res with the label)
                    _, pred = torch.max(outputs, 1)
                    total += labels.size(0)  # batch_size
                    correct_top1 += (pred == labels).sum().item()

                    # top 5
                    _, rank5 = outputs.topk(k=5,
                                            dim=1,
                                            largest=True,
                                            sorted=True)
                    rank5 = rank5.t()  # transpose dim 0 and 1
                    correct5 = rank5.eq(labels.view(1, -1).expand_as(rank5))

                    # for k in range(5):
                    k = 4
                    correct_k = correct5[:k + 1].reshape(-1).float().sum(
                        0, keepdim=True)
                    correct_top5 += correct_k.item()

            accuracy_top1 = correct_top1 / total
            accuracy_top5 = correct_top5 / total

            val_avg_loss = val_avg_loss / len(valid_loader)

            bar.set_postfix(Epoch=f"{epoch}/{opts.epoch}",
                            Valid_Avg_Loss=val_avg_loss,
                            Accuracy_Top1=accuracy_top1,
                            Accuracy_Top5=accuracy_top5)
            if vis is not None:
                vis.line(X=torch.ones((1, 3)) * epoch,
                         Y=torch.Tensor(
                             [accuracy_top1, accuracy_top5,
                              val_avg_loss]).unsqueeze(0),
                         update='append',
                         win='test_loss_acc',
                         opts=dict(x_label='epoch',
                                   y_label='test_loss and acc',
                                   title='test_loss and accuracy',
                                   legend=[
                                       'accuracy_top1', 'accuracy_top5',
                                       'avg_loss'
                                   ]))

            print(f"top-1 percentage: {accuracy_top1*100:.3f}%")
            print(f"top-5 percentage: {accuracy_top5*100:.3f}%")
            if scheduler:
                scheduler.step()
    return 0


def run(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler, num_epochs: int,
        local_rank: int):
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)

    df = generate_df()
    train_loader, valid_loader = get_dataloaders(df, Config["fold"],
                                                 GUIEDataset, alb_transforms)

    for epoch in range(1, num_epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler,
                                           train_loader, epoch)
        val_epoch_loss = eval_one_epoch(model,
                                        optimizer,
                                        valid_loader,
                                        epoch=epoch)
        if local_rank == 0:
            history['Train Loss'].append(train_epoch_loss)
            history['Valid Loss'].append(val_epoch_loss)

            # deep copy the model
            if val_epoch_loss <= best_epoch_loss:
                print(
                    f"{b_}Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})"
                )
                # best_epoch_loss = val_epoch_loss
                # best_model_wts = copy.deepcopy(model.state_dict())
                # PATH = os.path.join(
                #     Config['ckpt_dir'],
                #     f"Loss{best_epoch_loss:.4f}_epoch{epoch:.0f}.bin")
                # torch.save(model.state_dict(), PATH)
                # print(f"Model Saved{sr_}")
            print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60,
        (time_elapsed % 3600) % 60))
    print(f"Best Loss: {best_epoch_loss:.4f}")

    # load best model weights
    # model.load_state_dict(best_model_wts)

    # return model, history


if __name__ == "__main__":

    set_seed()
    config = ConfigType()

    # convert batchnorm layers to syncbatchnorm layers
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    df = generate_df()
    mp.spawn(main_worker,
             args=(
                 df,
                 config,
             ),
             nprocs=len(config.gpu_ids),
             join=True)
    torch.cuda.empty_cache()
