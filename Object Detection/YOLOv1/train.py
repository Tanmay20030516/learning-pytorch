import torch
import torchvision.transforms as transforms
# import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim

from model import YOLOv1
from dataset import VOCDataset
from loss import YOLOv1Loss
from utils import (iou, nms, mAP, cellboxes_to_boxes,
                   get_bboxes, plot_image, save_checkpoint, load_checkpoint)


seed = 123
torch.manual_seed(seed)

# Hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4  # originally 64, not enough ram in local machine's gpu
WEIGHT_DECAY = 0  # original paper uses lr scheduling, and a decay of 0.0005
NUM_EPOCHS = 5
NUM_CLASSES = 20
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"


# Creating a custom compose object that applies the transformations
class Compose(object):
    def __init__(self, transformations):
        self.transforms = transformations

    def __call__(self, img, bboxes):
        for t in self.transforms:
            # transforming only the images as bboxes are by default relative to image dims
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose(
    [
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ]
)


def train_loop(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())  # .item() converts single value tensor to simple number

    print(f"Mean loss: {sum(mean_loss) / len(mean_loss)}")


def main():
    # loading model and loss function
    model = YOLOv1(split_size=7, num_boxes_per_cell=2, num_classes=20, in_channels=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YOLOv1Loss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    # loading dataset and setting up the data loader
    train_dataset = VOCDataset(
        csv_file="data/100examples.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_dataset = VOCDataset(
        csv_file="data/test.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(NUM_EPOCHS):
        # visualizing
        # for x, y in train_loader:
        #
        #     x = x.to(DEVICE)
        #     for idx in range(8):
        #         bboxes = cellboxes_to_boxes(model(x))
        #         bboxes = nms(bboxes[idx], iou_threshold=0.5, prob_threshold=0.4, box_format="midpoint")
        #         plot_image(x[idx].permute(1, 2, 0).to("cpu"), bboxes)
        #
        #     import sys
        #     sys.exit()

        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, prob_threshold=0.4
        )

        mean_avg_precision = mAP(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=NUM_CLASSES
        )
        print(f"Train mAP: {mean_avg_precision}")

        # saving the model checkpoint
        if mean_avg_precision > 0.9:
            checkpoint = {
               "state_dict": model.state_dict(),
               "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            import time
            time.sleep(10)

        train_loop(train_loader, model, optimizer, loss_fn)


if __name__ == "__main__":
    main()
