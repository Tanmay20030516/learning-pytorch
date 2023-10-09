import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image


class VOCDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                #   int      /------float------\
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()  # since reading a string from a .txt file
                ]
                boxes.append([class_label, x, y, width, height])
        # extracting image from image path
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform is not None:
            # augment the boxes according to image augmentation
            image, boxes = self.transform(image, boxes)

        # convert the coordinates from relative to image, to relative to cell
        label_matrix = torch.zeros((self.S, self.S, self.C + 5*self.B))  # since this is target, only 1st C+5 will be used
        for box in boxes:
            class_label, x, y, w, h = box.tolist()  # box was a torch.tensor
            class_label = int(class_label)

            # getting index of cell to which object belongs
            i, j = int(self.S * y), int(self.S * x)

            # adjusting x, y values wrt cell dims            # |---- x and j
            x_cell, y_cell = self.S * x - j, self.S * y - i  # |
            # above is just origin shifting                  # y and i

            # adjusting w, h values wrt cell dims
            # width_pixels = width * self.image_width
            # cell_pixels = self.cell_width
            w_cell, h_cell = w * self.S, h * self.S  # just scaling w and h with cell dims

            # if we haven't seen current cell, mark it that it has an object
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1  # set that there exists object (pc = 1)

                # set the new coordinates wrt cell
                box_coordinates = torch.tensor([x_cell, y_cell, w_cell, h_cell])
                label_matrix[i, j, 21:25] = box_coordinates

                # one hot encode the class label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix