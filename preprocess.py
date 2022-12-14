import os
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tqdm import tqdm
from config import ConfigType

from preprocessing.data_config import DataConfigType
from preprocessing.imagenet1k import preprocess_ImageNet1k
from preprocessing.images130k import preprocess_Images130k
from preprocessing.google_landmark_2021 import preprocess_google_landmark_2021
from preprocessing.product10k import preprocess_Product10k
from preprocessing.clothing import preprocess_ClothingDataset
from preprocessing.hnm import preprocess_HnMFashionDataset
from preprocessing.ifood import preprocess_ifood
from preprocessing.met import preprocess_MET
from preprocessing.furniture_images import preprocess_furniture_images
from preprocessing.bonn_furniture import preprocess_BonnFurniture
from preprocessing.stanford_car import preprocess_StanfordCar


def get_dataframe_from_single_dataset(opt: DataConfigType) -> pd.DataFrame:
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
    elif data_name.lower() == "ifood":
        return preprocess_ifood(opt)
    elif data_name.lower() == "met":
        return preprocess_MET(opt)
    elif data_name.lower() == "furniture-images":
        return preprocess_furniture_images(opt)
    elif data_name.lower() == "bonn-furniture-styles-dataset":
        return preprocess_BonnFurniture(opt)
    elif data_name.lower() == "stanford-cars":
        return preprocess_StanfordCar(opt)
    else:
        raise ValueError(f"dataset {data_name} is not supported")

def downsample_dataset(df: pd.DataFrame, opt: DataConfigType) -> pd.DataFrame:
    if opt["downsample_rate"] > 1:
        label_count = df.groupby("label").count().rename({"file_path": "counts"}, axis=1)
        labels_all = label_count.index.to_list()
        downsampled_df = pd.DataFrame(columns=df.columns)
        for label in tqdm(labels_all, ncols=100, desc=f"downsampling {opt.data_name}"):
            n_labels = label_count.loc[label].counts
            n_ds = max(1, round(n_labels / opt["downsample_rate"]))
            sampled_recs = df[df["label"] == label].sample(n=n_ds)
            downsampled_df = pd.concat([downsampled_df, sampled_recs])
        downsampled_df.reset_index(drop=True, inplace=True)
        return downsampled_df
    else:
        return df

def preprocess_main(config: ConfigType) -> pd.DataFrame:
    # check whether the directories are valid
    for opt in config["data_config"]:
        if not os.path.exists(opt["data_dir"]):
            raise RuntimeError("data_dir not exist: " + opt["data_dir"])
    if not os.path.exists(config["save_path"]):
        raise RuntimeError(f"save_path not exist: " + config["save_path"])

    df_merged = pd.DataFrame(columns=["label", "file_path"])
    for opt in config["data_config"]:
        df = get_dataframe_from_single_dataset(opt)
        # select label_column (which is specified by opt) and "file_path"
        df = df[[opt["label_column"], "file_path"]]
        # change label_column to "label"
        df.rename({opt["label_column"]: "label"}, axis=1, inplace=True)
        # do downsampling
        df = downsample_dataset(df, opt)
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
    for label_id, label in enumerate(encoder.classes_):
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
