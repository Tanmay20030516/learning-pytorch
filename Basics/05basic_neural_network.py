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


# 1. Creating our Neural network
class DNN(nn.Module):  # subclassing from nn.Module
    """
    Here we define the layers of the network. We create two fully connected layers

    Parameters:
        input_size: the size of the input, in this case 784 (28x28)
        num_classes: the number of classes we want to predict, in this case 10 (0-9)

    """
    def __init__(self, input_size, num_classes):
        super(DNN, self).__init__()
        self.fc_layer1 = nn.Linear(input_size, 200)
        self.fc_layer2 = nn.Linear(200, 100)
        self.fc_layer3 = nn.Linear(100, 50)
        self.fc_layer4 = nn.Linear(50, num_classes)

    def forward(self, x):
        """
        x here is the mnist images, and we run it through fully connected layers that we created above.
        we also add a ReLU activation function in between

        Parameters:
            x: mnist images

        Returns:
            out: the output of the network
        """
        x = F.relu(self.fc_layer1(x))
        x = F.relu(self.fc_layer2(x))
        x = F.relu(self.fc_layer3(x))
        x = self.fc_layer4(x)
        return x


# 2. Set device cuda for GPU if it's available otherwise run on the CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Hyperparameters
INPUT_SIZE = 784  # 28*28
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 7

# 4. Data Loading
train_dataset = datasets.MNIST(
    root='dataset/', train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root='dataset/', train=False, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True
)

# 5. Network Initialisation
model = DNN(input_size=INPUT_SIZE, num_classes=NUM_CLASSES).to(DEVICE)

# 6. Setting up loss and optimiser
loss_fn = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 7. Network training
print("Training started...")
for epoch in range(NUM_EPOCHS):
    for batch_idx, (X, y) in enumerate(tqdm(train_loader)):
        # load to cuda (if possible)
        X = X.to(device=DEVICE)
        y = y.to(device=DEVICE)

        # reshaping
        X = X.reshape(X.shape[0], -1)  # X of shape (batch, num_channels, height, width)

        # forward pass
        predicted = model(X)
        loss = loss_fn(predicted, y)

        # backward pass
        optimiser.zero_grad()  # set the grads to zero for new batch
        loss.backward()

        # optimisation step
        optimiser.step()


# 8. Accuracy measure
def accuracy_score(loader, model):
    """
    Check accuracy of our trained model given a loader and a model

    Parameters:
        loader: torch.utils.data.DataLoader
            A loader for the dataset you want to check accuracy on
        model: nn.Module
            The model you want to check accuracy on

    Returns:
        acc: float
            The accuracy of the model on the dataset given by the loader
    """
    num_correct = 0
    num_examples = 0
    model.eval()  # evaluation mode

    with torch.no_grad():  # no need to track grads while testing
        for data, target in loader:
            # process data
            data = data.to(device=DEVICE)
            target = target.to(device=DEVICE)
            data = data.reshape(data.shape[0], -1)

            # forward pass (for prediction)
            val = model(data)  # val.shape = (batch_size, predictions)
            # so we want the index of maximum score
            _, predictions = val.max(dim=1)

            # assessing accuracy
            num_correct += (predictions == target).sum()
            num_examples += predictions.size(dim=0)

    model.train()
    return num_correct/num_examples


# 9. Final evaluation
print(f"Accuracy on train: {accuracy_score(train_loader, model)*100:.2f}%")
print(f"Accuracy on test: {accuracy_score(test_loader, model)*100:.2f}%")
