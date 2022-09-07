# generate dataframe for training/evaluation
# img_id | file_path | label_id
#    0     data/a.jpg   N0100213 (imagenet)

import os
from tqdm import tqdm
import json
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import re

from config import Config
from imagenet1k_classes import IMAGENET2012_CLASSES

# TODO: wrap label encoding with a integrated function

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
├── LOC_sample_submission.csv
├── LOC_synset_mapping.txt
├── LOC_train_solution.csv
└── LOC_val_solution.csv
"""
def preprocess_ImageNet1k() -> pd.DataFrame:
    data_dir = os.path.join(Config.data_dir, "ILSVRC", "Data", "CLS-LOC", "train")
    meta_dir = os.path.join(Config.data_dir, "ILSVRC", "ImageSets", "CLS-LOC", "train_cls.txt")

    # validate data integrity
    n_class = 0
    for dir_name in os.listdir(data_dir):
        if re.search("n[0-9]{7,}", dir_name):
            assert dir_name in IMAGENET2012_CLASSES.keys()
            n_class += 1
    assert n_class == len(IMAGENET2012_CLASSES)
    df = pd.read_csv(meta_dir, sep=" ", header=None, names=["label_and_image_name", "count"])

    # read train metadata only
    # TODO: read val & test splits as well (currently we split the train into train & val again)
    # DF columns : label_code | image_name | label
    # ex) n1440764 | n01440764_11151.JPEG | tench, Tinca tinca
    tqdm.pandas(ncols=100, desc="obtaining label_code")
    df["label_code"] = df.progress_apply(
        lambda rec: rec['label_and_image_name'].split('/')[0],
        axis=1
        # label_code example: n01440764
    )
    tqdm.pandas(ncols=100, desc="obtaining image_name")
    df["image_name"] = df.progress_apply(
        lambda rec: rec['label_and_image_name'].split('/')[1] + ".JPEG",
        axis=1
        # assume images are given in JPEG format
    )
    tqdm.pandas(ncols=100, desc="obtaining label")
    df["label"] = df.progress_apply(
        lambda rec: IMAGENET2012_CLASSES[rec['label_code']],
        axis=1
    )

    # remove unnecessary columns
    df = df.drop(["label_and_image_name", "count"], axis=1)

    # add file_path column
    tqdm.pandas(ncols=100, desc="obtaining file_path")
    df['file_path'] = df.progress_apply(
        lambda rec: os.path.join(data_dir, rec['label_code'], rec['image_name']),
        axis=1
        # file_path example: ../ImageNet1k/n01440764/n01440764_11151.JPEG
    )

    # encode label
    encoder = LabelEncoder()
    df["label_id"] = encoder.fit_transform(df['label'])

    class_mappings = {}
    class_inv_mappings = {}
    labels = df['label'].unique()
    for label in labels:
        label_id = list(df[df['label'] == label]['label_id'])[0]
        class_mappings[label] = label_id
        class_inv_mappings[label_id] = label

    data_dir = Config.data_dir
    class_mapping_path = os.path.join(data_dir, "class_mapping.json")
    class_inv_mapping_path = os.path.join(data_dir, "class_inv_mapping.json")

    with open(class_mapping_path, "wt") as f:
        json.dump(class_mappings, f)

    with open(class_inv_mapping_path, "wt") as f:
        json.dump(class_inv_mappings, f)

    return df
    

# ------------------------------------------------------

"""
use dataset from https://www.kaggle.com/datasets/rhtsingh/130k-images-512x512-universal-image-embeddings
follow the file structure as follows

{DATA_ROOT_DIR}
├── apparel
├── artwork
├── cars
├── dishes
├── furniture
├── illustrations
├── landmark
├── meme
├── packaged
├── storefronts
├── toys
└── train.csv
"""

def read_data_Images130k() -> pd.DataFrame:
    data_dir = Config.data_dir
    df_path = os.path.join(data_dir, "train.csv")
    df = pd.read_csv(df_path)
    df['file_path'] = df.apply(
        lambda rec: os.path.join(data_dir, rec['label'], rec['image_name']),
        axis=1
        # file_path example: ../Images130k/apparel/image0000.jpg
    )
    return df


def preprocess_label_Images130k(df: pd.DataFrame) -> pd.DataFrame:
    #label encoding
    encoder = LabelEncoder()
    df['label_id'] = encoder.fit_transform(df['label'])

    class_mappings = {}
    class_inv_mappings = {}
    labels = df['label'].unique()
    for label in labels:
        label_id = list(df[df['label'] == label]['label_id'])[0]
        class_mappings[label] = label_id
        class_inv_mappings[label_id] = label

    data_dir = Config.data_dir
    class_mapping_path = os.path.join(data_dir, "class_mapping.json")
    class_inv_mapping_path = os.path.join(data_dir, "class_inv_mapping.json")

    with open(class_mapping_path, "wt") as f:
        json.dump(class_mappings, f)

    with open(class_inv_mapping_path, "wt") as f:
        json.dump(class_inv_mappings, f)

    return df


def preprocess_Images130k() -> pd.DataFrame:
    df = read_data_Images130k()
    df = preprocess_label_Images130k(df)
    return df

# ------------------------------------------------------

"""
use dataset from https://www.kaggle.com/competitions/landmark-retrieval-2021/data
follow the file structure as follows

{DATA_ROOT_DIR}
├── index
├── sample_submission.csv
├── test
├── train
│   ├── 0
│   │   ├── 0
│   │   │   ├── 0
:   :   :   :
│   │   │   └── f
│   │   ├── 1
│   │   │   ├── 0
:   :   :   :
│   │   │   └── f
:   :   :
│   └── f
│       ├── 0
│       │   ├── 0
:       :   :
│       │   └── f
:       :   
│       └── f
│           ├── 0
:           :
│           └── f
└── train.csv

(test, index, sample_submission.csv are not required)
"""

def preprocess_google_landmark_2021() -> pd.DataFrame:
    data_dir = os.path.join(Config.data_dir, "train")
    meta_path = os.path.join(Config.data_dir, "train.csv")

    # read train metadata
    df = pd.read_csv(meta_path, sep=",", header=1, names=["image_name", "landmark_id"])
    tqdm.pandas(ncols=100, desc="obtaining label")
    df["label"] = df.progress_apply(
        lambda rec: f"landmark_{rec['landmark_id']}",
        axis=1
        # label example: landmark_1
    )
    tqdm.pandas(ncols=100, desc="obtaining file_path")
    df['file_path'] = df.progress_apply(
        lambda rec: os.path.join(
            data_dir,
            rec['image_name'][0],
            rec['image_name'][1],
            rec['image_name'][2],
            rec['image_name'] + ".jpg"
        ),
        axis=1
        # file_path example: ../landmark-retrieval-2021/train/1/7/6/17660ef415d37059.jpg
    )

    # remove unnecessary columns
    df = df.drop(["image_name", "landmark_id"], axis=1)

    # encode label
    encoder = LabelEncoder()
    df["label_id"] = encoder.fit_transform(df['label'])

    class_mappings = {}
    class_inv_mappings = {}
    labels = df['label'].unique()
    for label in labels:
        label_id = list(df[df['label'] == label]['label_id'])[0]
        class_mappings[label] = label_id
        class_inv_mappings[label_id] = label

    data_dir = Config.data_dir
    class_mapping_path = os.path.join(data_dir, "class_mapping.json")
    class_inv_mapping_path = os.path.join(data_dir, "class_inv_mapping.json")

    with open(class_mapping_path, "wt") as f:
        json.dump(class_mappings, f)

    with open(class_inv_mapping_path, "wt") as f:
        json.dump(class_inv_mappings, f)

    return df


# ------------------------------------------------------


def preprocess_main() -> pd.DataFrame:
    print(f"Processing {Config.data_name} started")
    if Config.data_name.lower() == "images130k":
        return preprocess_Images130k()
    elif Config.data_name.lower() == "imagenet1k":
        return preprocess_ImageNet1k()
    elif Config.data_name.lower() == "google-landmark-2021":
        return preprocess_google_landmark_2021()
    else:
        raise ValueError(f"dataset {Config.data_name} is not supported")


if __name__ == "__main__":
    df = preprocess_main()
    print(df)
