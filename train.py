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
import wandb
import visdom
from config import ConfigType
from sklearn.model_selection import StratifiedKFold
from dataset import GUIEDataset, alb_transforms
from loss import cross_entropy_loss
from model import GUIEModel, CLIPModel
from preprocess import preprocess_main
from utils import init_for_distributed, setup_for_distributed

##TODO:: different model output for training/inference (https://www.kaggle.com/code/motono0223/guie-clip-tensorflow-train-example/notebook)
##TODO:: best model? (selected metric by user)
##TODO:: train_one_epoch/ evaluate function


def setup_run(args: ConfigType):
    if args.log_all:
        run = wandb.init(
            project=args.project,
            group="DDP",
        )
    else:
        if args.log_rank == 0:
            run = wandb.init(project=args.project, )
        else:
            run = None

    return run


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
    return df


def generate_df(opt: ConfigType) -> pd.DataFrame:
    df = preprocess_main(opt)
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
    # split into train/val
    fold = opts.fold
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    # if there is a remainder in train dataset,
    # drop the record whose label is the most frequently appeared
    # to prevent ValueError of BatchNorm, which requires more than two samples per batch
    if len(df_train) % opts.train_sample_per_gpu:
        rm_idx = df_train[df_train.label == df_train.label.value_counts().idxmax()].index[-1]
        df_train.drop(rm_idx, axis=0, inplace=True)
        df_train = df_train.reset_index(drop=True)
    # build dataset, sampler and loader
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


def main_worker(rank, df: pd.DataFrame, opts: ConfigType, run):
    local_gpu_id = init_for_distributed(rank, opts)
    is_master = local_gpu_id == opts.gpu_ids[0]

    # wandb log
    do_log = run is not None

    # data loader
    loaders_dict = get_dataloaders(df, alb_transforms, opts)
    train_loader, train_sampler = loaders_dict["train"]
    valid_loader, _ = loaders_dict["valid"]

    # model creation and loading
    n_cls = len(df["label_id"].unique())
    if "clip" in opts.model_type.lower():
        print(f"Creating CLIPModel for {n_cls} classes") # sanity check
        model = CLIPModel(opts, n_cls)
    elif "swin" in opts.model_type.lower():
        print(f"Creating GUIEModel for {n_cls} classes") # sanity check
        model = GUIEModel(opts, n_cls)
    else:
        raise ValueError(f"{opts.model_type} is not supported")
    model = model.cuda(local_gpu_id)
    if opts.load_from is not None:
        ckpt_data = torch.load(opts.load_from, map_location=next(model.parameters()).device)
        model.load_model(ckpt_data['model_state_dict'])
        print(f"load model from {opts.load_from} is completed")
    model = DDP(module=model, device_ids=[local_gpu_id])

    # watch gradients for rank0
    if is_master and do_log:
        run.watch(model)

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
        train_avg_loss = 0
        bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=150)
        for step, (images, labels) in bar:
            images = images.to(local_gpu_id)
            labels = labels.to(local_gpu_id)
            outputs = model(images, labels)

            # update
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_avg_loss += loss.item()

            # retreive lr
            lr = optimizer.param_groups[0]['lr']
            if scheduler:
                scheduler.step()

            # time
            toc = time.time()

            # visualization
            bar.set_description(
                f"Epoch [{epoch + 1}/{opts.epoch}], Loss: {loss.item():.4f}, "
                f"LR: {lr:.3e}, Time: {toc-tic:.2f}")
            if (step % opts.vis_step == 0
                    or step == len(train_loader) - 1) and opts.rank == 0:
                if do_log:
                    run.log({"train_loss": loss.item()})
                    run.log({"LR": lr})

        if do_log:
            run.log({
                "epoch": epoch,
                "train_avg_loss": train_avg_loss / len(train_loader),
            })

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

        # evaluation
        if opts.rank == 0:
            model.eval()

            val_avg_loss = 0
            correct_top1 = 0
            correct_top5 = 0
            total = 0

            with torch.no_grad():
                bar = tqdm(enumerate(valid_loader), total=len(valid_loader), \
                    ncols=150, desc="validation in progress")
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

            if do_log:
                run.log({
                    "epoch": epoch,
                    "val_avg_loss": val_avg_loss,
                    "top-1 accuracy": accuracy_top1,
                    "top-5 accuracy": accuracy_top5
                })

            print(f"top-1 percentage: {accuracy_top1*100:.3f}%")
            print(f"top-5 percentage: {accuracy_top5*100:.3f}%")
    return 0


##TODO:: deprecated
# def run(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
#         scheduler: torch.optim.lr_scheduler._LRScheduler, num_epochs: int,
#         local_rank: int):
#     start = time.time()
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_epoch_loss = np.inf
#     history = defaultdict(list)

#     df = generate_df()
#     train_loader, valid_loader = get_dataloaders(df, Config["fold"],
#                                                  GUIEDataset, alb_transforms)

#     for epoch in range(1, num_epochs + 1):
#         train_loader.sampler.set_epoch(epoch)
#         train_epoch_loss = train_one_epoch(model, optimizer, scheduler,
#                                            train_loader, epoch)
#         val_epoch_loss = eval_one_epoch(model,
#                                         optimizer,
#                                         valid_loader,
#                                         epoch=epoch)
#         if local_rank == 0:
#             history['Train Loss'].append(train_epoch_loss)
#             history['Valid Loss'].append(val_epoch_loss)

#             # deep copy the model
#             if val_epoch_loss <= best_epoch_loss:
#                 print(
#                     f"{b_}Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})"
#                 )
#                 # best_epoch_loss = val_epoch_loss
#                 # best_model_wts = copy.deepcopy(model.state_dict())
#                 # PATH = os.path.join(
#                 #     Config['ckpt_dir'],
#                 #     f"Loss{best_epoch_loss:.4f}_epoch{epoch:.0f}.bin")
#                 # torch.save(model.state_dict(), PATH)
#                 # print(f"Model Saved{sr_}")
#             print()

#     end = time.time()
#     time_elapsed = end - start
#     print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
#         time_elapsed // 3600, (time_elapsed % 3600) // 60,
#         (time_elapsed % 3600) % 60))
#     print(f"Best Loss: {best_epoch_loss:.4f}")

#     # load best model weights
#     # model.load_state_dict(best_model_wts)

#     # return model, history

if __name__ == "__main__":

    # reproducibility
    set_seed()

    # read user config
    config = ConfigType()
    # TODO: export and import config
    # TODO: create new checkpoint directory based on current time and date 
    #       whenever the train starts

    # set wandb run
    run = setup_run(config)

    # generate dataframe with preprocessing
    df = generate_df(config)
    # TODO: show data statistics

    # multiprocessing for multi-gpu training
    mp.spawn(main_worker,
             args=(
                 df,
                 config,
                 run,
             ),
             nprocs=len(config.gpu_ids),
             join=True)

    #emtpy cuda memory after training
    torch.cuda.empty_cache()

    # finish wandb process
    wandb.finish()
