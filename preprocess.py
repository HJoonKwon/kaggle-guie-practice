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

def preprocess_ImageNet1k() -> pd.DataFrame:
    data_dir = Config.data_dir

    # validate data integrity
    n_class = 0
    for dir_name in os.listdir(data_dir):
        if re.search("n[0-9]{7,}", dir_name):
            assert dir_name in IMAGENET2012_CLASSES.keys()
            n_class += 1
    assert n_class == len(IMAGENET2012_CLASSES)

    # create an empty DataFrame
    df = pd.DataFrame(columns=["label_code", "image_name", "label"])
    # and traverse every files in the dataset
    for dir_name in tqdm(sorted(os.listdir(data_dir)), ncols=100):
        if not re.search("n[0-9]{7,}", dir_name):
            continue
        for fn in os.listdir(os.path.join(data_dir, dir_name)):
            if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                # DF columns : label_code | image_name | label
                # ex) n1440764 | n01440764_11151.JPEG | tench, Tinca tinca
                df.loc[len(df)] = dir_name, fn, IMAGENET2012_CLASSES[dir_name]

    # add file_path column
    df['file_path'] = df.apply(
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

def preprocess_main() -> pd.DataFrame:
    print(f"Processing {Config.data_name} started")
    if Config.data_name.lower() == "images130k":
        return preprocess_Images130k()
    elif Config.data_name.lower() == "imagenet1k":
        return preprocess_ImageNet1k()
    else:
        raise ValueError(f"dataset {Config.data_name} is not supported")


if __name__ == "__main__":
    df = preprocess_main()
    print(df)
