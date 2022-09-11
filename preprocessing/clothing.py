import os
from tqdm import tqdm
import pandas as pd

from preprocessing.data_config import DataConfigType

"""
use dataset from https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full
follow the file structure as follows

{DATA_ROOT_DIR}
├── images.csv
├── images_compressed
└── images_original

return:
    DataFrame of columns: label | file_path | supercategory
    ex) T-shirt | ../clothing-dataset-full/images_original/ea7b6656-3f84-4eb3-9099-23e623fc1018.jpg | apparel

list of labels (total 17 categories):
    'T-Shirt', 'Shoes', 'Shorts', 'Shirt', 'Pants',
    'Skirt', 'Top', 'Outwear', 'Dress', 'Body',
    'Longsleeve', 'Undershirt', 'Hat', 'Polo', 'Blouse',
    'Hoodie', 'Blazer'
"""

def preprocess_ClothingDataset(opt: DataConfigType) -> pd.DataFrame:
    data_dir = os.path.join(opt["data_dir"], "images_original")
    meta_dir = os.path.join(opt["data_dir"], "images.csv")

    # read metadata
    df = pd.read_csv(meta_dir)

    # remove unnecessary records
    df.drop(
        df[
            (df["label"] == "Not sure") |
            (df["label"] == "Other") |
            (df["label"] == "Skip")
        ].index,
        inplace=True
    )
    df.reset_index(inplace=True)

    # add file path column
    tqdm.pandas(ncols=100, desc="obtaining file_path (1/1)")
    df["file_path"] = df.progress_apply(
        lambda rec: os.path.join(data_dir, rec["image"] + ".jpg"),
        axis=1
    )

    # add supercategory column
    df["supercategory"] = "apparel"

    # remove unnecessary column
    df.drop(
        ["index", "image", "sender_id", "kids"],
        axis=1,
        inplace=True
    )

    return df
