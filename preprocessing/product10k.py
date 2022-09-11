import os
from tqdm import tqdm
import pandas as pd

from preprocessing.data_config import DataConfigType

"""
use dataset from https://www.kaggle.com/competitions/products-10k/data
follow the file structure as follows

{DATA_ROOT_DIR}
├── test
├── train
└── train.csv

return:
    DataFrame of columns: label | product_group | file_path | supercategory
    ex) product_0 | product_group_0 | .../Product10k/train/1.jpg | packaged
    (It is guaranteed that label is unique across the product_group)
"""

def preprocess_Product10k(opt: DataConfigType) -> pd.DataFrame:
    data_dir = os.path.join(opt["data_dir"], "train")
    meta_dir = os.path.join(opt["data_dir"], "train.csv")

    # read train metadata
    df = pd.read_csv(meta_dir)

    # check whether class is unique regardless of the group in whicn the record is included
    class_list = df["class"].unique().tolist()
    for class_label in class_list:
        assert len(df["group"][df["class"] == class_label].unique().tolist()) == 1

    # renaming label from 'class' column
    tqdm.pandas(ncols=100, desc="obtaining label (1/3)")
    df["label"] = df.progress_apply(
        lambda rec: "product_" + str(rec["class"]),
        axis=1
    )
    
    # renaming group from 'group' column
    tqdm.pandas(ncols=100, desc="obtaining group (2/3)")
    df["product_group"] = df.progress_apply(
        lambda rec: "product_group_" + str(rec["group"]),
        axis=1
    )

    # add file_path column
    tqdm.pandas(ncols=100, desc="obtaining file_path (3/3)")
    df['file_path'] = df.progress_apply(
        lambda rec: os.path.join(data_dir, rec["name"]),
        axis=1
    )

    # remove unnecessary columns
    df.drop(
        ["name", "class", "group"],
        axis=1,
        inplace=True
    )

    return df
