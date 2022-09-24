import os
from tqdm import tqdm
import pandas as pd

from preprocessing.data_config import DataConfigType
from preprocessing.hnm_classes import HNM_UNIFIED_CLASS_MAPPING

"""
use dataset from https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data
follow the file structure as follows

{DATA_ROOT_DIR}
├── articles.csv
└── images

return:
    DataFrame of columns: product_type_name | label | file_path | supercategory
    ex) Vest top | Top | .../h-and-m-personalized-fashion-recommendations/images/010/0108775015.jpg | apparel

list of labels (total 20 valid categories):
    (17 categories are in common with the labels of clothing dataset)
    'T-Shirt', 'Shoes', 'Shorts', 'Shirt', 'Pants',
    'Skirt', 'Top', 'Outwear', 'Dress', 'Body',
    'Longsleeve', 'Undershirt', 'Hat', 'Polo', 'Blouse',
    'Hoodie', 'Blazer'
    
    (three categories are added compared to clothing dataset)
    'Underwear', 'Accessories', 'Bag'

    (and there are two more invalid categories)
    'Unclassified': included in the supercategory "apparel", but not sure in which label included
    'None': not even included in the supercategory "apparel"

    TODO: use Unclassified if user selects supercategory as label
    TODO: give other supercategory to None
"""

def preprocess_HnMFashionDataset(opt: DataConfigType) -> pd.DataFrame:
    data_dir = os.path.join(opt["data_dir"], "images")
    meta_dir = os.path.join(opt["data_dir"], "articles.csv")

    # read metadata (use only "article_id" and "product_type_name" columns)
    df = pd.read_csv(meta_dir, usecols=["article_id", "product_type_name"])

    # map existing classes into unified classes
    tqdm.pandas(ncols=100, desc="obtaining label (1/3)")
    df["label"] = df.progress_apply(
        lambda rec: HNM_UNIFIED_CLASS_MAPPING[rec['product_type_name']],
        axis=1
    )

    # remove invalid records
    df.drop(
        df[
            (df.label == "None") |
            (df.label == "Unclassified")
        ].index,
        inplace=True
    )
    df.reset_index(inplace=True, drop=True)

    # add file_path column
    tqdm.pandas(ncols=100, desc="obtaining file_path (2/3)")
    df["file_path"] = df.progress_apply(
        lambda rec: os.path.join(data_dir, "0" + str(rec["article_id"])[:2], "0" + str(rec["article_id"]) + ".jpg"),
        axis=1
    )

    # remove records whose images are missing
    tqdm.pandas(ncols=100, desc="removing missing images (3/3)")
    df["file_exist"] = df.progress_apply(
        lambda rec: os.path.exists(rec["file_path"]),
        axis=1
    )
    df.drop(
        df[df.file_exist == False].index,
        inplace=True
    )
    df.reset_index(inplace=True, drop=True)

    # add supercategory column
    df["supercategory"] = "apparel"

    # remove unnecessary columns
    df.drop(
        ["article_id", "file_exist"],
        axis=1,
        inplace=True
    )

    return df
