import os
from tqdm import tqdm
import pandas as pd

from preprocessing.data_config import DataConfigType

"""
use dataset from https://www.kaggle.com/datasets/rhtsingh/130k-images-512x512-universal-image-embeddings
follow the file structure as follows

{DATA_ROOT_DIR}
├── apparel
├── artwork
├── cars
├── dishes
├── furniture
├── illustrations
├── landmark
├── meme
├── packaged
├── storefronts
├── toys
└── train.csv

return:
    DataFrame of columns: label | file_path
    ex) apparel | .../Images130k/apparel/image000.jpg
"""

def preprocess_Images130k(opt: DataConfigType) -> pd.DataFrame:
    data_dir = opt.data_dir
    meta_dir = os.path.join(data_dir, "train.csv")

    # read meta data
    df = pd.read_csv(meta_dir)
    
    # add file_path column
    tqdm.pandas(ncols=100, desc="obtaining file_path (1/1)")
    df['file_path'] = df.progress_apply(
        lambda rec: os.path.join(data_dir, rec['label'], rec['image_name']),
        axis=1
        # file_path example: ../Images130k/apparel/image0000.jpg
    )

    # remove unnecessary columns
    df.drop(
        ["image_name"],
        axis=1,
        inplace=True
    )

    return df