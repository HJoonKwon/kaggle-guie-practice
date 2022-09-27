import os
from tqdm import tqdm
import pandas as pd

from preprocessing.data_config import DataConfigType
from scipy.io import loadmat

"""
use dataset from http://ai.stanford.edu/~jkrause/cars/car_dataset.html
follow the file structure as follows

{DATA_ROOT_DIR}
├── cars_train
└── devkit
    ├── README.txt
    ├── cars_meta.mat
    ├── cars_test_annos.mat
    ├── cars_train_annos.mat
    ├── eval_train.m
    └── train_perfect_preds.txt

return:
    DataFrame of columns: label | file_path | supercategory
    ex) Audi R8 Coupe 2012 | 00001.jpg | cars
"""

def preprocess_StanfordCar(opt: DataConfigType) -> pd.DataFrame:
    # load index-label mapping
    label_mapping_path = os.path.join(opt["data_dir"], "devkit", "cars_meta.mat")
    label_mapping_np = loadmat(label_mapping_path, matlab_compatible=False, simplify_cells=True, chars_as_strings=True)
    idx_label_map = {}
    for idx, label in enumerate(label_mapping_np["class_names"].tolist()):
        idx_label_map[idx + 1] = label

    # load metadata
    meta_path = os.path.join(opt["data_dir"], "devkit", "cars_train_annos.mat")
    meta_np = loadmat(meta_path, matlab_compatible=False, simplify_cells=True, chars_as_strings=True)
    df = pd.DataFrame(meta_np["annotations"])

    # map existing classes into human-readable label
    tqdm.pandas(ncols=100, desc="obtaining label (1/2)")
    df["label"] = df.progress_apply(
        lambda rec: idx_label_map[rec["class"]],
        axis=1
    )

    # add file_path column
    tqdm.pandas(ncols=100, desc="obtaining file_path (2/2)")
    df["file_path"] = df.progress_apply(
        lambda rec: os.path.join(opt["data_dir"], "cars_train", rec["fname"]),
        axis=1
    )

    # add supercategory column
    df["supercategory"] = "cars"

    # remove unnecessary columns
    df.drop(
        ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "class", "fname"],
        axis=1,
        inplace=True
    )

    return df
