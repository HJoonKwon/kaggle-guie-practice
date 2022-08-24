import torch

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
    learning_rate = 1e-4
    scheduler = "CosineAnnealingLR"
    min_lr = 1e-6
    T_max = 500
    weight_decay = 1e-6
    n_fold=5
    n_accumulate=1
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    num_epochs = 10
    train_batch_size = 4
    valid_batch_size = 4
    num_gpus = 4
    num_classes = 11

    model_name = "swin_large_patch4_window12_384_in22k" #"convnext"
    img_size = sizes[model_name]

    data_dir = '/media/volume4/130k-512x512-guie'
    fold = 1
    ckpt_dir = '/media/volume4/130k-512x512-guie/ckpts'

    # ArcFace Hyperparameters
    s = 30.0
    m =0.50
    ls_eps = 0.0
    easy_margin = False

    def __getitem__(self, key):
        return getattr(self, key, None)


Config = ConfigType()
