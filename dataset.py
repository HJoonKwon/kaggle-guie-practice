import torch
from torch.utils.data import Dataset
import albumentations as A
import cv2

class GUIEDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.file_paths = df["file_path"].values
        self.labels = df["label_id"].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_paths[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[index]

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return {
            "image": img,
            "label": torch.Tensor(label, torch.long)
        }
