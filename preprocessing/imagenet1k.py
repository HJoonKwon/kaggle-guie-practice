import os
from tqdm import tqdm
import pandas as pd
import re

from preprocessing.data_config import DataConfigType
from preprocessing.imagenet1k_classes import IMAGENET2012_CLASSES

"""
use dataset from https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data
follow the file structure as follows

{DATA_ROOT_DIR}
├── ILSVRC
│   ├── Annotations
│   │   └── CLS-LOC
│   │       ├── train
│   │       └── val
│   ├── Data
│   │   └── CLS-LOC
│   │       ├── test
│   │       ├── train
│   │       └── val
│   └── ImageSets
│       └── CLS-LOC
│           ├── test.txt
│           ├── train_cls.txt
│           ├── train_loc.txt
│           └── val.txt
├── LOC_synset_mapping.txt
├── LOC_train_solution.csv
└── LOC_val_solution.csv

return:
    DataFrame of columns: label_code | label | file_path
    ex) n1440764 | tench, Tinca tinca | .../ImageNet1k/n01440764/n01440764_11151.JPEG
"""
def preprocess_ImageNet1k(opt: DataConfigType) -> pd.DataFrame:
    data_dir = os.path.join(opt.data_dir, "ILSVRC", "Data", "CLS-LOC", "train")
    meta_dir = os.path.join(opt.data_dir, "ILSVRC", "ImageSets", "CLS-LOC", "train_cls.txt")

    # validate data integrity
    n_class = 0
    for dir_name in os.listdir(data_dir):
        if re.search("n[0-9]{7,}", dir_name):
            assert dir_name in IMAGENET2012_CLASSES.keys()
            n_class += 1
    assert n_class == len(IMAGENET2012_CLASSES)

    # read train metadata only
    df = pd.read_csv(meta_dir, sep=" ", header=None, names=["label_and_image_name", "count"])

    # TODO: read val & test splits as well (currently we split the train into train & val again)
    # add label_code column
    tqdm.pandas(ncols=100, desc="obtaining label_code (1/3)")
    df["label_code"] = df.progress_apply(
        lambda rec: rec['label_and_image_name'].split('/')[0],
        axis=1
    )
    # add label column
    tqdm.pandas(ncols=100, desc="obtaining label (2/3)")
    df["label"] = df.progress_apply(
        lambda rec: IMAGENET2012_CLASSES[rec['label_code']],
        axis=1
    )
    # add file_path column
    tqdm.pandas(ncols=100, desc="obtaining file_path (3/3)")
    df['file_path'] = df.progress_apply(
        lambda rec: os.path.join(data_dir, rec['label_code'], rec['label_and_image_name'].split('/')[1] + ".JPEG"),
        axis=1
    )

    # remove unnecessary columns
    df.drop(
        ["label_and_image_name", "count"],
        axis=1,
        inplace=True
    )

    return df
    