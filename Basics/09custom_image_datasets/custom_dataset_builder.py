import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CatsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):  # to get the length of annotations dataframe
        return len(self.annotations)

    def __getitem__(self, index):  # to get a particular example with its label
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        img = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            img = self.transform(img)

        return (img, y_label)





