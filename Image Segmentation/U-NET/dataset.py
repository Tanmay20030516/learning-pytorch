import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class CarvanaDataset(Dataset):
    """
    creating the carvana dataset for our model
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_path = os.path.join(self.image_dir, self.images[item])
        mask_path = os.path.join(self.mask_dir, self.images[item].replace('.jpg', '_mask.gif'))
        # reading the image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0  # setting all bright pixels to 1
        # 1.0, as a positive label, as later on we'll use sigmoid function

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
