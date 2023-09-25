# IMPORTS
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision
from torch.utils.data import DataLoader
from custom_dataset_builder import CatsDataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HYPERPARAMETERS
in_channels = 3
num_classes = 2
learning_rate = 3e-4
batch_size = 32
num_epochs = 10

# DATASET CREATION
dataset = CatsDataset(
    csv_file="cats.csv",  # contains the image name and corresponding label
    root_dir="cats",  # the folder containing the images
    transform=transforms.ToTensor()
)

# DATASET LOADING
# randomly splits into 8 and 2, although actual dataset is not just of 10 images
train_set, test_set = torch.utils.data.random_split(dataset, [8, 2])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# !!! make sure that all the images are of same size !!!

model = torchvision.models.googlenet(weights="DEFAULT")

for param in model.parameters():  # parameter freezing
    param.requires_grad = False

# modifying last layer
model.fc = nn.Linear(in_features=1024, out_features=num_classes)

# MODEL LOADING
model = model.to(device)

# LOSS FUNCTION AND OPTIMIZER
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

# TRAINING
for epoch in range(num_epochs):
    losses = []
    for batch_idx, (X, y) in enumerate(tqdm(train_loader)):
        X = X.to(device=device)
        y = y.to(device=device)
        # forward
        prediction = model(X)
        loss = loss_fn(prediction, y)
        losses.append(loss.item())
        # backward
        optimizer.zero_grad()
        loss.backward()
        # optimization step
        optimizer.step()

    print(f"Cost at epoch {epoch+1} is {(sum(losses) / len(losses)):.4f}")


# TESTING
def accuracy_score(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # sets model to evaluation mode

    with torch.no_grad():  # not keeping track of gradients
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(dim=1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(dim=0)

        print(f"Got {num_correct} / {num_samples} with accuracy {(float(num_correct)/float(num_samples))*100:.2f}%")

    model.train()  # sets model back to training mode (usage of features like batch norm, dropout)


print("Checking accuracy on train set...")
accuracy_score(train_loader, model)

print("Checking accuracy on test set")
accuracy_score(test_loader, model)


