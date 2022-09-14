import os
from tqdm import tqdm
import pandas as pd

from preprocessing.data_config import DataConfigType

"""
use dataset from https://cvml.comp.nus.edu.sg/furniture/
follow the file structure as follows

{DATA_ROOT_DIR}
├── houzz
│   ├── beds
│   ├── chairs
│   ├── dressers
│   ├── lamps
│   ├── sofas
│   └── tables
└── metadata
    ├── beds.txt
    ├── chairs.txt
    ├── dressers.txt
    ├── lamps.txt
    ├── sofas.txt
    └── tables.txt

return:
    DataFrame of columns: label | style | file_path | supercategory
    ex) beds | Asian | .../Bonn_Furniture_Styles_Dataset/houzz/beds/Asian/19726asian-daybeds.jpg | furniture
"""

def preprocess_BonnFurniture(opt: DataConfigType) -> pd.DataFrame:
    meta_root_dir = os.path.join(opt["data_dir"], "metadata")

    if len(os.listdir(meta_root_dir)) != 6:
        raise RuntimeError(f"Number of metadata file is not 6. check {meta_root_dir}")

    # gather all metadata
    df_merged = pd.DataFrame(columns=["label", "style", "image_name"])
    for meta_path in tqdm(os.listdir(meta_root_dir), ncols=100, desc="obtaining metadata (1/2)"):
        label_name = meta_path.split(".")[0]
        df = pd.read_csv(
            os.path.join(meta_root_dir, meta_path),
            sep="\t",
            header=None,
            names=["style", "image_name", "NaN", "metadata"]
        )

        df["label"] = label_name
        df.drop(
            ["NaN", "metadata"],
            axis=1,
            inplace=True
        )

        df_merged = pd.concat([df_merged, df])

    # add file_path column
    tqdm.pandas(ncols=100, desc="obtaining file_path (2/2)")
    df_merged["file_path"] = df_merged.progress_apply(
        lambda rec: os.path.join(opt["data_dir"], rec["image_name"]),
        axis=1
    )

    # add supercategory column
    df_merged["supercategory"] = "furniture"

    # remove unnecessary columns
    df_merged.drop(
        ["image_name"],
        axis=1,
        inplace=True
    )

    return df_merged
