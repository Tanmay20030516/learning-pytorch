import os
import albumentations as A
import numpy as np
from PIL import Image
tr = A.Compose([A.Resize(width=1920, height=1080)])
images_paths = os.listdir('images/')
i = 1
for img in images_paths:
    img_pth = 'images/'+img
    image = Image.open(img_pth)
    image = np.array(image)
    augmentations = tr(image=image)
    aug_img = augmentations["image"]
    # print(aug_img)
    aug_img = Image.fromarray(aug_img)
    aug_img.save(f"img{i}.jpeg")
    i = i + 1


