import os
import json
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from config import Config

from preprocessing.data_config import DataConfigType
from preprocessing.imagenet1k import preprocess_ImageNet1k
from preprocessing.images130k import preprocess_Images130k
from preprocessing.google_landmark_2021 import preprocess_google_landmark_2021
from preprocessing.product10k import preprocess_Product10k
from preprocessing.clothing import preprocess_ClothingDataset
from preprocessing.hnm import preprocess_HnMFashionDataset


def get_dataframe_from_single_dataset(opt: DataConfigType) ->pd.DataFrame:
    print(f"Processing {opt.data_name} started")
    if opt.data_name.lower() == "images130k":
        return preprocess_Images130k(opt)
    elif opt.data_name.lower() == "imagenet1k":
        return preprocess_ImageNet1k(opt)
    elif opt.data_name.lower() == "google-landmark-2021":
        return preprocess_google_landmark_2021(opt)
    elif opt.data_name.lower() == "product10k":
        return preprocess_Product10k(opt)
    elif opt.data_name.lower() == "clothing-dataset":
        return preprocess_ClothingDataset(opt)
    elif opt.data_name.lower() == "hnm-fashion-dataset":
        return preprocess_HnMFashionDataset(opt)
    else:
        raise ValueError(f"dataset {opt.data_name} is not supported")


def preprocess_main() -> pd.DataFrame:
    # TODO: express data configuration of 'Config' in terms of DataConfigType
    opt = DataConfigType()
    opt.data_name = Config.data_name
    opt.data_dir = Config.data_dir
    
    df = get_dataframe_from_single_dataset(opt)

    # encode label
    encoder = LabelEncoder()
    df["label_id"] = encoder.fit_transform(df['label'])

    # save class mapping to the ckpt directory
    label_mappings = {}
    label_inv_mappings = {}
    labels = df['label'].unique()
    for label in labels:
        label_id = list(df[df['label'] == label]['label_id'])[0]
        label_mappings[label] = label_id
        label_inv_mappings[label_id] = label

    save_path = Config.save_path
    label_mapping_path = os.path.join(save_path, "label_mapping.json")
    label_inv_mapping_path = os.path.join(save_path, "label_inv_mapping.json")

    print("exporting label - label_id mapping...")
    with open(label_mapping_path, "wt") as f:
        json.dump(label_mappings, f)

    with open(label_inv_mapping_path, "wt") as f:
        json.dump(label_inv_mappings, f)
    print("export completed!")

    return df

if __name__ == "__main__":
    df = preprocess_main()
    print(df)
