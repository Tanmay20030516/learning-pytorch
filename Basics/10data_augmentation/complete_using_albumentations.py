# Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from model_builder import CNN, model_train
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from tqdm import tqdm


# Augmentation pipeline
class ImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        super(ImageFolder, self).__init__()
        self.data = []  # [(image1, label1), (image2, label2), ..., (image10, label10), ...]
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = os.listdir(root_dir)  # all folders in root directory
        # setting up the data list
        for idx, name in enumerate(self.class_names):
            file_paths = os.listdir(os.path.join(root_dir, name))  # lists all files in path root_dir/name
            self.data += (list(zip(file_paths, [idx] * len(file_paths))))  # [9]*8 -> [9, 9, 9, 9, 9, 9, 9, 9]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_file, label = self.data[item]
        root_and_dir = os.path.join(self.root_dir, self.class_names[label], img_file)  # complete path of image/file
        # print(root_and_dir)
        image = cv2.cvtColor(cv2.imread(root_and_dir), cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return (image, label)


# Transformations
transform = A.Compose(
    [
        # A.Resize(width=1920, height=1080),
        # A.RandomCrop(width=1280, height=720),
        A.Rotate(limit=20, p=0.75, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.8),
        A.VerticalFlip(p=0.01),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.7),
        A.OneOf([
            A.Blur(blur_limit=4, p=0.6),  # has a probability of 0.8*0.6 to happen
            A.ColorJitter(p=0.7)  # has a probability of 0.8*0.7 to happen
        ], p=0.8),
        A.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1],
            max_pixel_value=255,  # the scaled and shifted pixel value multiplied by this value
        ),
        ToTensorV2(),
    ]
)

# Setting up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
in_channels = 3
num_classes = 2

# Hyperparameters
batch_size = 1
learning_rate = 1e-3
num_epochs = 10

# Load the dataset
train_dataset = ImageFolder(root_dir="cat1_cat2", transform=transform)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

# Loading our network
model = CNN(in_channels=in_channels, num_classes=num_classes)
model = model.to(device=device)
print(summary(model, input_size=(3, 256, 256)))
print("\n")

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# Training
model = model_train(
    model=model,
    device=device,
    loader=train_loader,
    num_epochs=num_epochs,
    loss_function=loss_fn,
    optimizer=optimizer
)


# Testing
def accuracy_score(loader, model):
    # if loader.dataset.train:
    #     print("Checking accuracy on train set...\n")
    # else:
    #     print("Checking accuracy on test set...\n")

    num_correct = 0
    num_samples = 0
    model.eval()  # set the model on evaluation mode

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            y_predicted = model(X)
            _, predicted_class = y_predicted.max(dim=1)
            num_correct += (predicted_class == y).sum()
            num_samples += y_predicted.size(dim=0)

    print(f"Accuracy: {(num_correct / num_samples) * 100:.3f}%")
    model.train()


accuracy_score(train_loader, model)

