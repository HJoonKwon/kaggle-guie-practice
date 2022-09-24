import os
from tqdm import tqdm
import pandas as pd
import json

from preprocessing.data_config import DataConfigType

"""
use dataset from https://www.kaggle.com/datasets/dschettler8845/the-met-dataset
follow the file structure as follows

{DATA_ROOT_DIR}
├── MET
│   └── MET [224408 entries exceeds filelimit, not opening dir]
├── descriptor_models
│   ├── r18IN_con-syn+real-closest_descriptors.pkl
│   └── r18SWSL_con-syn+real-closest_descriptors.pkl
└── ground_truth
    ├── MET_database.json
    ├── mini_MET_database.json
    ├── testset.json
    └── valset.json

return:
    DataFrame of columns: label | file_path | supercategory
    ex) artifact_770869 | .../the-met-dataset/MET/MET/770869/0.jpg | artwork
"""

def preprocess_MET(opt: DataConfigType) -> pd.DataFrame:
    data_dir = opt["data_dir"]
    meta_dir = os.path.join(data_dir, "ground_truth", "MET_database.json")

    # read meta data
    with open(meta_dir, "rt") as f:
        df = pd.DataFrame(json.load(f))
    
    # rename label column
    tqdm.pandas(ncols=100, desc="obtaining label (1/2)")
    df["label"] = df.progress_apply(
        lambda rec: "artifact_" + str(rec["id"]),
        axis=1
    )

    # add file_path column
    tqdm.pandas(ncols=100, desc="obtaining file_path (2/2)")
    df["file_path"] = df.progress_apply(
        lambda rec: os.path.join(data_dir, "MET", rec["path"]),
        axis=1
    )

    # add supercategory column
    df["supercategory"] = "artwork"

    # remove unnecessary columns
    df.drop(
        ["id", "path"],
        axis=1,
        inplace=True
    )

    return df