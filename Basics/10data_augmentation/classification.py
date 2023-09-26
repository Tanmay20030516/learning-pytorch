import cv2
import albumentations as A
import numpy as np
from utils import plot_examples
from PIL import Image  # reading image, can be done via cv2 also
from tqdm import tqdm

# reading image
image = Image.open("images/elon.jpeg")

# setting up the augmentation pipeline
transform = A.Compose([
    A.Resize(width=1920, height=1080),
    A.RandomCrop(width=1280, height=720),
    A.Rotate(limit=20, p=0.75, border_mode=cv2.BORDER_CONSTANT),
    A.HorizontalFlip(p=0.8),
    A.VerticalFlip(p=0.01),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.7),
    A.OneOf([
        A.Blur(blur_limit=4, p=0.6),  # has a probability of 0.8*0.6 to happen
        A.ColorJitter(p=0.7)  # has a probability of 0.8*0.7 to happen
    ], p=0.8)
])

# making augmentations
image_list = [image]
image = np.array(image)
for i in tqdm(range(15)):  # try 15 augmentations
    augmentations = transform(image=image)  # creates a dict
    augmented_image = augmentations["image"]  # retrieve the augmented image
    image_list.append(augmented_image)

# visualising the augmented images
plot_examples(image_list)
