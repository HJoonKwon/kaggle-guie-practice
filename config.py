from typing import List, Optional
from preprocessing.data_config import DataConfigType

sizes = {
    # EfficientNet
    'b0': (256, 224),
    'b1': (256, 240),
    'b2': (288, 288),
    'b3': (320, 300),
    'b4': (384, 380),
    'b5': (489, 456),
    'b6': (561, 528),
    'b7': (633, 600),

    # DINO
    'dino_vitb8': (256, 224),
    'dino_vits8': (256, 224),
    'dino_vitb16': (256, 224),
    'dino_vits16': (256, 224),

    # ConvNext
    'convnext_large_in22ft1k': (224, 224),
    'convnext_xlarge_in22ft1k': (224, 224),
    'convnext_base_384_in22ft1k': (384, 384),
    'convnext_large_384_in22ft1k': (384, 384),
    'convnext_xlarge_384_in22ft1k': (384, 384),
    'convnext_base_in22k': (224, 224),
    'convnext_large_in22k': (224, 224),
    'convnext_xlarge_in22k': (224, 224),

    # Swin Transformer
    'swin_tiny_patch4_window7_224': (224, 224),
    'swin_base_patch4_window7_224': (224, 224),
    'swin_base_patch4_window12_384': (384, 384),
    'swin_large_patch4_window7_224': (224, 224),
    'swin_small_patch4_window7_224': (224, 224),
    'swin_large_patch4_window12_384': (384, 384),
    'swin_base_patch4_window7_224_in22k': (224, 224),
    'swin_base_patch4_window12_384_in22k': (384, 384),
    'swin_large_patch4_window7_224_in22k': (224, 224),
    'swin_large_patch4_window12_384_in22k': (384, 384),
}


class ConfigType:

    # optimizer setting
    optim_name: str = "SGD"  #"Adam"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    momentum: float = 0.9

    # learning rate scheduler
    scheduler = "CosineAnnealingLR"
    min_lr: float = 1e-6
    T_max: int = 500
    T_0: int = 5

    # ex. StepLR
    step_size: int = 30
    gamma: float = 0.1

    # data split (GroupKFold)
    n_fold: int = 5
    fold: int = 1

    # traning setting
    load_from: Optional[str] = None
    n_accumulate: int = 1
    start_epoch: int = 0
    epoch: int = 5

    # dataset configuration
    # List of availabel data_name is in preprocessing.data_config
    data_config: List[DataConfigType] = [
        DataConfigType(**{
            "data_name": "Images130k",
            "data_dir": "/media/dounkim/KDU/Individual_Study/kaggle/google-universal-image-embedding/Images130k"
        }),
        DataConfigType(**{
            "data_name": "HnM-Fashion-Dataset",
            "data_dir": "/media/dounkim/KDU/Individual_Study/kaggle/google-universal-image-embedding/h-and-m-personalized-fashion-recommendations",
            "label_column": "supercategory"
        }),
        DataConfigType(**{
            "data_name": "Google-Landmark-2021",
            "data_dir": "/media/dounkim/KDU/Individual_Study/kaggle/google-universal-image-embedding/landmark-retrieval-2021",
            "label_column": "supercategory"
        })
    ]

    # model setting
    # num_classes: int = NUM_CLASSES[data_name] TODO: compute num_classes automatically
    embedding_size: int = 256
    model_name: str = "swin_large_patch4_window12_384_in22k"  #"convnext"
    img_size: tuple = sizes[model_name]
    # ArcFace Hyperparameters
    s: float = 30.0
    m: float = 0.30
    ls_eps: float = 0.0
    easy_margin = False

    # directories
    save_path: str = '/media/volume4/130k-512x512-guie/ckpts'
    save_file_name: str = f'{model_name}_images130k'

    # multi-gpu setting
    gpu_ids: List[str] = ["0", "1", "2", "3"]
    num_workers_per_gpu: int = 4
    train_sample_per_gpu: int = 4
    valid_sample_per_gpu: int = 4
    rank: int = 0
    port: int = 2022

    # wandb logging
    vis_step: int = 100
    log_all: bool = False
    log_rank: int = 0
    project: str = "kaggle-guie"

    def __getitem__(self, key):
        return getattr(self, key, None)


Config = ConfigType()
