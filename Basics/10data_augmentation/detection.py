import cv2
import albumentations as A
# import numpy as np
from utils import plot_examples
# from PIL import Image
from tqdm import tqdm


image = cv2.cvtColor(cv2.imread("images/cat.jpg"), cv2.COLOR_BGR2RGB)
bboxes = [[13, 170, 224, 410]]  # pascal_voc format (x_min, y_min, x_max, y_max)

transform = A.Compose(
    [
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
    ],
    bbox_params=A.BboxParams(  # bboxes to be updated according to augmented image
        format="pascal_voc",
        min_area=2048,  # the minimum bbox area on the image needed for that box to be valid
        min_visibility=0.4,  # %age of entire image being covered by bbox
        label_fields=()
    )
)

images_list = [image]  # we don't need to convert image to numpy, as read using cv2
saved_bboxes = [bboxes[0]]

for i in tqdm(range(25)):
    augmentations = transform(image=image, bboxes=bboxes)
    augmented_image = augmentations["image"]
    if len(augmentations["bboxes"]) == 0:
        continue
    images_list.append(augmented_image)
    saved_bboxes.append(augmentations["bboxes"][0])

plot_examples(images_list, saved_bboxes)
