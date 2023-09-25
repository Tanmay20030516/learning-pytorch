# Imports
import torch
import torch.nn.functional as F  # parameterless functions, (some of the activation functions)
import torchvision.datasets as datasets  # standard dataset
import torchvision.transforms as transforms  # transformations to perform on dataset for augmentation
from torch import optim  # optimisers
from torch import nn  # all nn modules
from torch.utils.data import (
    DataLoader,
)  # for dataset management
from tqdm import tqdm  # progress bar customisation


# Simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,  # number of kernels
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )
        self.fc1 = nn.Linear(
            in_features=32*7*7,  # 28 -> 14 -> 7
            out_features=num_classes,
        )

    def forward(self, x):
        # 1st conv block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # 2nd conv block
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # Flattening followed by fully connected layers
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channels = 1  # MNIST digits dataset is grayscale
num_classes = 10  # 0-9
learning_rate = 3e-4  # karpathy's constant
batch_size = 64
num_epochs = 5

# Load Data
train_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root="dataset/", train=False, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=True
)
print("Data loaded...")

# Initialize network
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
print("Training started...")
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward pass
        scores = model(data)
        loss = loss_fn(scores, targets)

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # optimization step
        optimizer.step()


# Check accuracy on training & test to see how good our model
def accuracy_score(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # evaluation mode

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            # print(scores.shape)  # (64, 10)
            _, predictions = scores.max(1)  # so we want the index of maximum score
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


print(f"Accuracy on training set: {accuracy_score(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {accuracy_score(test_loader, model)*100:.2f}")


