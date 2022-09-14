import os
from tqdm import tqdm
import pandas as pd
import re

from preprocessing.data_config import DataConfigType

"""
use dataset from https://www.kaggle.com/datasets/lasaljaywardena/furniture-images-dataset
follow the file structure as follows

{DATA_ROOT_DIR}
├── furniture_data_img.csv
└── furniture_images
    └── furniture_images

return:
    DataFrame of columns: label | file_path | desc | supercategory
    ex) Bed / bedroom item | .../furnitue_images/1634011559093_Bed Room Set for sale.jpg | bed room set | furniture

list of labels (total 9 categories)
    'Other', 'Bed / bedroom item', 'TV / stereo', 'Storage', 'Table / chair',
    'Sofa / living room item', 'Antique / art', 'Lighting', 'Textiles / decoration'
"""

def preprocess_furniture_images(opt: DataConfigType) -> pd.DataFrame:
    data_dir = opt["data_dir"]
    meta_dir = os.path.join(data_dir, "furniture_data_img.csv")

    # read meta data
    df = pd.read_csv(meta_dir)

    # rename label, desc column
    df.rename(
        {
            "Label": "desc",
            "Furniture_Type": "label"
        },
        axis=1,
        inplace=True
    )

    # add file_path column
    tqdm.pandas(ncols=100, desc="obtaining file_path (1/1)")
    df["file_path"] = df.progress_apply(
        # Note) use naive add operator instead of os.path.join because of white spaces in "Image_File"
        lambda rec: os.path.join(data_dir, "furniture_images") + rec["Image_File"],
        axis=1
    )

    # add supercategory column
    df["supercategory"] = "furniture"

    # remove unnecessary columns
    df.drop(
        ["Image_File"],
        axis=1,
        inplace=True
    )

    return df