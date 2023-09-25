import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchsummary import summary
from tqdm import tqdm


# Defining our CNN model
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.globalmaxpool = nn.MaxPool2d(kernel_size=32)
        self.fc1 = nn.Linear(in_features=1024, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)
        self.do = nn.Dropout()

    def forward(self, x):
        # conv block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # conv block 2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        # conv block 3
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        # global max pooling (alternative to flattening)
        x = self.globalmaxpool(x)
        # fully connected layers
        x = x.reshape(x.size(dim=0), -1)  # flatten the tensor fully before passing to fc layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.do(x)
        x = self.fc2(x)
        return x


# Setting up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_classes = 10
learning_rate = 3e-4
batch_size = 8
num_epochs = 1

# Initialise the model
model = CNN(in_channels=3, num_classes=num_classes)
model.to(device=device)
# print(summary(model, input_size=(3, 256, 256)))  # printing model summary just like model.summary() in tf.keras

# Transformations (image augmentations)
my_transforms = transforms.Compose(
    [
        # transforms.Resize((256, 256)),  # resizes to (256, 256)
        transforms.Resize((270, 270)),  # resizes to (270, 270) so random cropping can be done
        transforms.RandomCrop((256, 256)),  # random cropping to (256, 256) from (270, 270)
        transforms.ColorJitter(brightness=0.6),
        transforms.RandomRotation(degrees=33),  # random rotation from (-33, +33) degrees
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.05),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        # the mean and std dev are calculated across each color channel, the pixel values updated as
        # pixel_new = (pixel_old - mean) / std
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])  # although these values don't do anything
    ]
)

# Loading train data
train_dataset = datasets.CIFAR10(root="Basics/dataset/", train=True, transform=my_transforms, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Model training
for epoch in range(num_epochs):
    losses = []
    for batch_idx, (X_train, y_train) in enumerate(tqdm(train_loader)):
        X_train = X_train.to(device=device)
        y_train = y_train.to(device=device)
        # forward pass
        prediction = model(X_train)
        loss = loss_fn(prediction, y_train)
        losses.append(loss.item())
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # optimization step
        optimizer.step()

    print(f"Loss at epoch {epoch+1}: {(sum(losses)/len(losses)):.4f}")


# Model testing
def accuracy_score(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on train set...\n")
    else:
        print("Checking accuracy on test set...\n")

    num_correct = 0
    num_samples = 0
    model.eval()  # set the model on evaluation mode

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)
            _, pred_class = y_pred.max(dim=1)
            num_correct += (pred_class == y).sum()
            num_samples += y_pred.size(dim=0)

    print(f"Accuracy: {(num_correct/num_samples)*100:.3f}%")
    model.train()


accuracy_score(train_loader, model)
