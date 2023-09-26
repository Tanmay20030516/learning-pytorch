import cv2
import albumentations as A
import numpy as np
from utils import plot_examples
from PIL import Image  # reading image, can be done via cv2 also
from tqdm import tqdm

# reading image and its corresponding masks
image = Image.open("images/elon.jpeg")
mask = Image.open("images/mask.jpeg")
mask2 = Image.open("images/second_mask.jpeg")

# setting up the transforms
transform = A.Compose(
    [
        A.Resize(width=1920, height=1080),
        A.RandomCrop(width=1280, height=720),
        A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.ColorJitter(p=0.5),
        ], p=1.0),
    ]
)

# image and mask augmentations
image_list = [image]
image = np.array(image)
mask = np.array(mask)
mask2 = np.array(mask2)

for i in tqdm(range(4)):
    augment = transform(image=image, masks=[mask, mask2])
    augment_img = augment["image"]
    augment_mask = augment["masks"]
    image_list.append(augment_img)
    image_list.append(augment_mask[0])
    image_list.append(augment_mask[1])

plot_examples(image_list)

