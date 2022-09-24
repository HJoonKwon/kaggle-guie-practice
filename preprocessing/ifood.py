import os
from tqdm import tqdm
import pandas as pd

from preprocessing.data_config import DataConfigType
from preprocessing.ifood_classes import get_ifood_2019_class_dict

"""
use dataset from https://www.kaggle.com/competitions/ifood-2019-fgvc6/data
follow the file structure as follows

{DATA_ROOT_DIR}
├── class_list.txt
├── test_set
├── train_labels.csv
├── train_set
├── val_labels.csv
└── val_set

return:
    DataFrame of columns: label_code | label | file_path | supercategory
    ex) 247 | eccles_cake | .../ifood/val_set/val_010323.jpg | dishes
"""

def preprocess_ifood(opt: DataConfigType) -> pd.DataFrame:
    data_dir = opt["data_dir"]
    train_meta_dir = os.path.join(data_dir, "train_labels.csv")
    val_meta_dir = os.path.join(data_dir, "val_labels.csv")
    IFOOD_2019_CLASS_DICT = get_ifood_2019_class_dict(data_dir)

    # read train meta data
    train_df = pd.read_csv(train_meta_dir)
    # read val meta data
    val_df = pd.read_csv(val_meta_dir)

    # rename label column
    train_df.rename(
        {"label": "label_code"},
        axis=1,
        inplace=True
    )
    val_df.rename(
        {"label": "label_code"},
        axis=1,
        inplace=True
    )

    # add label column
    tqdm.pandas(ncols=100, desc="obtaining train label (1/4)")
    train_df["label"] = train_df.progress_apply(
        lambda rec: IFOOD_2019_CLASS_DICT[rec["label_code"]],
        axis=1
    )
    tqdm.pandas(ncols=100, desc="obtaining val label (2/4)")
    val_df["label"] = val_df.progress_apply(
        lambda rec: IFOOD_2019_CLASS_DICT[rec["label_code"]],
        axis=1
    )

    # add file_path column
    tqdm.pandas(ncols=100, desc="obtaining train file_path (3/4)")
    train_df["file_path"] = train_df.progress_apply(
        lambda rec: os.path.join(data_dir, "train_set", rec["img_name"]),
        axis=1
        # file_path example: ../ifood/train_set/train_101733.jpg
    )
    tqdm.pandas(ncols=100, desc="obtaining val file_path (4/4)")
    val_df["file_path"] = val_df.progress_apply(
        lambda rec: os.path.join(data_dir, "val_set", rec["img_name"]),
        axis=1
        # file_path example: ../ifood/val_set/val_010323.jpg
    )

    # merge train & val DataFrame
    df_merged = pd.concat([train_df, val_df])

    # add supercategory column
    df_merged["supercategory"] = "dishes"

    # remove unnecessary columns
    df_merged.drop(
        ["img_name"],
        axis=1,
        inplace=True
    )

    return df_merged