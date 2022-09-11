import os
from tqdm import tqdm
import pandas as pd

from preprocessing.data_config import DataConfigType

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

return:
    DataFrame of columns: label | file_path | supercategory
    ex) landmark_0 | .../landmark-retrieval-2021/train/9/2/b/92b6290d571448f6.jpg | landmark
"""

def preprocess_google_landmark_2021(opt: DataConfigType) -> pd.DataFrame:
    data_dir = os.path.join(opt["data_dir"], "train")
    meta_path = os.path.join(opt["data_dir"], "train.csv")

    # read train metadata
    df = pd.read_csv(meta_path, sep=",", header=1, names=["image_name", "landmark_id"])

    # add label column
    tqdm.pandas(ncols=100, desc="obtaining label (1/2)")
    df["label"] = df.progress_apply(
        lambda rec: f"landmark_{rec['landmark_id']}",
        axis=1
        # label example: landmark_1
    )

    # add file_path column
    tqdm.pandas(ncols=100, desc="obtaining file_path (2/2)")
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

    # add supercategory column
    df["supercategory"] = "landmark"

    # remove unnecessary columns
    df.drop(
        ["image_name", "landmark_id"],
        axis=1,
        inplace=True
    )

    return df
