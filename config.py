import torch
class Config:
    learning_rate = 1e-4
    scheduler = "CosineAnnealingLR"
    min_lr = 1e-6
    T_max = 500
    weight_decay = 1e-6
    n_fold=5
    n_accumulate=1
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    num_epochs = 20
    train_batch_size = 32
    valid_batch_size = 64
    num_gpus = 4
    num_classes = 10000

    model_name = "vit_clip" #"convnext"

    # ArcFace Hyperparameters
    s = 30.0
    m =0.50
    ls_eps = 0.0
    easy_margin = False

