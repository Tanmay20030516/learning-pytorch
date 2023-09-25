# Imports
import torch
import torch.nn.functional as F  # parameterless functions, (some of the activation functions)
import torchvision.datasets as datasets  # standard dataset
import torchvision.models
import torchvision.transforms as transforms  # transformations to perform on dataset for augmentation
from torch import optim  # optimisers
from torch import nn  # all nn modules
from torch.utils.data import (
    DataLoader,
)  # for dataset management
from tqdm import tqdm  # progress bar customisation

# Setting up the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 256
NUM_EPOCHS = 15

# Load a pretrained model
model = torchvision.models.vgg16(weights="DEFAULT")  # pre-trained weights for vgg16 model
# print(model)

# x = torch.rand((64, 3, 32, 32))  # where (3, 32, 32) is shape of cifar-10 images
# output is of type (batch_size, out_channels, height, width)
# print(model.features(x).shape)  # (batch_size, 512, 1, 1)

# If we want to fine tune our model (i.e. we modify and train last few layers)
for param in model.parameters():
    param.requires_grad = False  # freeze the weights

# model has three parts,
# 1. feature - feature extractors (conv and pool blocks)
# 2. avgpool - global average pooling (alternative to flattening)
# 3. classifier - classification part
# Fine tune the last layers; the modification in avgpool and classifier part makes these layers as trainable
model.avgpool = nn.Identity()
model.classifier = nn.Sequential(
    nn.Linear(512, 100),
    nn.ReLU(),
    nn.Linear(100, NUM_CLASSES)
)
# print(model)  # fine-tuned model

# Moving the model to cuda (if available)
model.to(DEVICE)

# Loading the CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root="dataset", train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = datasets.CIFAR10(root="dataset", train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Network training
for epoch in range(NUM_EPOCHS):
    losses = []
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data = data.to(device=DEVICE)
        target = target.to(device=DEVICE)

        # forward pass
        pred = model(data)
        loss = loss_fn(pred, target)
        losses.append(loss.item())

        # back pass
        optimizer.zero_grad()  # reset the optimizer
        loss.backward()  # calculate the gradients

        # optimization step
        optimizer.step()

    print(f"Cost at epoch {epoch + 1} is {sum(losses) / len(losses):.5f}\n")


# Accuracy of the model
def accuracy_score(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data...")
    else:
        print("Checking accuracy on testing data...")

    num_correct = 0
    num_samples = 0

    with torch.no_grad():  # not calculating gradients
        for dataa, label in loader:
            dataa = dataa.to(device=DEVICE)
            label = label.to(device=DEVICE)

            prediction = model(dataa)
            _, predictions = prediction.max(dim=1)
            num_correct += (predictions == label).sum()
            num_samples += predictions.sum(dim=0)

        print(f"Got accuracy of {(float(num_correct) / float(num_samples)) * 100:.3f}%")

    model.train()


accuracy_score(train_loader, model)
accuracy_score(test_loader, model)
