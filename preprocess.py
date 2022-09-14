import os
import json
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from config import ConfigType

from preprocessing.data_config import DataConfigType
from preprocessing.imagenet1k import preprocess_ImageNet1k
from preprocessing.images130k import preprocess_Images130k
from preprocessing.google_landmark_2021 import preprocess_google_landmark_2021
from preprocessing.product10k import preprocess_Product10k
from preprocessing.clothing import preprocess_ClothingDataset
from preprocessing.hnm import preprocess_HnMFashionDataset


def get_dataframe_from_single_dataset(opt: DataConfigType) ->pd.DataFrame:
    data_name = opt["data_name"]
    print(f"Processing {data_name} started")
    if data_name.lower() == "images130k":
        return preprocess_Images130k(opt)
    elif data_name.lower() == "imagenet1k":
        return preprocess_ImageNet1k(opt)
    elif data_name.lower() == "google-landmark-2021":
        return preprocess_google_landmark_2021(opt)
    elif data_name.lower() == "product10k":
        return preprocess_Product10k(opt)
    elif data_name.lower() == "clothing-dataset":
        return preprocess_ClothingDataset(opt)
    elif data_name.lower() == "hnm-fashion-dataset":
        return preprocess_HnMFashionDataset(opt)
    else:
        raise ValueError(f"dataset {data_name} is not supported")


def preprocess_main(config: ConfigType) -> pd.DataFrame:
    df_merged = pd.DataFrame(columns=["label", "file_path"])
    for opt in config["data_config"]:
        df = get_dataframe_from_single_dataset(opt)
        # select label_column (which is specified by opt) and "file_path"
        df = df[[opt["label_column"], "file_path"]]
        # change label_column to "label"
        df.rename({opt["label_column"]: "label"}, axis=1, inplace=True)
        # and merge
        df_merged = pd.concat([df_merged, df])
    df_merged.reset_index(drop=True, inplace=True)
    print(f"Total {len(df_merged['label'].unique())} classes as a result of preprocessing")

    # encode label
    encoder = LabelEncoder()
    df_merged["label_id"] = encoder.fit_transform(df_merged['label'])

    # save class mapping to the ckpt directory
    label_mappings = {}
    label_inv_mappings = {}
    labels = df_merged['label'].unique()
    for label in labels:
        label_id = list(df_merged[df_merged['label'] == label]['label_id'])[0]
        label_mappings[label] = label_id
        label_inv_mappings[label_id] = label

    save_path = config.save_path
    label_mapping_path = os.path.join(save_path, "label_mapping.json")
    label_inv_mapping_path = os.path.join(save_path, "label_inv_mapping.json")

    print("exporting label - label_id mapping...")
    with open(label_mapping_path, "wt") as f:
        json.dump(label_mappings, f)

    with open(label_inv_mapping_path, "wt") as f:
        json.dump(label_inv_mappings, f)
    print("export completed!")

    return df_merged

if __name__ == "__main__":
    from config import Config
    df = preprocess_main(Config)
    print(df)
