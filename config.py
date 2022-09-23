from typing import List, Optional
from preprocessing.data_config import DataConfigType
import os

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

DATA_COMMON = "/data1/dukim/kaggle/"

clip_models_path = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
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
            "data_name": "Bonn-Furniture-Styles-Dataset",
            "data_dir": os.path.join(DATA_COMMON, "Bonn_Furniture_Styles_Dataset"),
            "label_column": "supercategory"
        }),
        DataConfigType(**{
            "data_name": "Furniture-Images",
            "data_dir": os.path.join(DATA_COMMON, "furniture-images-dataset"),
            "label_column": "supercategory"
        }),
        DataConfigType(**{
            "data_name": "MET",
            "data_dir": os.path.join(DATA_COMMON, "the-met-dataset"),
            "label_column": "supercategory",
            "downsample_rate": 2
        }),
        DataConfigType(**{
            "data_name": "Images130k",
            "data_dir": os.path.join(DATA_COMMON, "Images130k")
        }),
        DataConfigType(**{
            "data_name": "Clothing-Dataset",
            "data_dir": os.path.join(DATA_COMMON, "clothing-dataset-full"),
            "label_column": "supercategory"
        }),
        DataConfigType(**{
            "data_name": "HnM-Fashion-Dataset",
            "data_dir": os.path.join(DATA_COMMON, "h-and-m-personalized-fashion-recommendations"),
            "label_column": "supercategory"
        }),
        DataConfigType(**{
            "data_name": "Product10k",
            "data_dir": os.path.join(DATA_COMMON, "Product10k"),
            "label_column": "supercategory"
        }),
        DataConfigType(**{
            "data_name": "Google-Landmark-2021",
            "data_dir": os.path.join(DATA_COMMON, "landmark-retrieval-2021"),
            "label_column": "supercategory",
            "downsample_rate": 8
        }),
        DataConfigType(**{
            "data_name": "iFood",
            "data_dir": os.path.join(DATA_COMMON, "ifood"),
            "label_column": "supercategory"
        })
    ]

    # model setting
    raw_embedding_size: int = 256
    model_name: str = "swin_large_patch4_window12_384_in22k"  #"convnext"
    clip_version: str = 'ViT-L/14@336px'
    img_size: tuple = sizes[model_name]
    clip_output_size: int = 768
    output_size: int = 64
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
