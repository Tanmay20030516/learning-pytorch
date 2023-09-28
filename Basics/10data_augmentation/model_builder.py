# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F


# Creating our network
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.globalmaxpool = nn.MaxPool2d(kernel_size=16)
        self.fc1 = nn.Linear(in_features=256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x):
        # conv and pool layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        # global max pooling layer
        x = self.globalmaxpool(x)
        # fc layers
        x = x.reshape(x.size(dim=0), -1)  # flatten the tensor fully before passing to fc layers
        x = self.fc1(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        return x


# def model_summary(mod : class, in_channels, num_classes, input_size):
#     model = mod(in_channels, num_classes)
#     return summary(model, input_size=input_size)


def model_train(model, device, loader, num_epochs, loss_function, optimizer):
    """
    :param model: model to train
    :param device: "cpu" or "cuda"
    :param loader: data loader
    :param num_epochs: number of epochs
    :param loss_function: loss function
    :param optimizer: optimizer like Adam, SGD
    :return: model - trained model
    """
    for epoch in range(num_epochs):
        losses = []
        for batch_idx, (X, y) in enumerate(loader):
            X_train = X.to(device=device)
            y_train = y.to(device=device)
            # forward
            prediction = model(X_train)
            loss = loss_function(prediction, y_train)
            losses.append(loss.item())
            # calculate gradients
            optimizer.zero_grad()
            loss.backward()
            # optimization step (weight update)
            optimizer.step()
        print(f"\nLoss after epoch {epoch + 1}: {(sum(losses) / len(losses)) * 100:.5f}")
    return model
