# TensorBoard on PyTorch [Documentation]: https://pytorch.org/docs/stable/tensorboard.html

# Imports
import torch
import torchvision
import torch.nn as nn  # NN modules
import torch.optim as optim  # optimizers
import torch.nn.functional as F  # parameter less functions
import torchvision.datasets as datasets  # standard datasets
import torchvision.transforms as transforms  # transformations
from torch.utils.data import DataLoader  # dataset management
from torch.utils.tensorboard import SummaryWriter  # making tensorboard summary


# CNN model
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.globalmaxpool = nn.MaxPool2d(kernel_size=3)
        self.fc1 = nn.Linear(in_features=128, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        # conv block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        # conv block 2
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        # conv block 3
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.pool(x)
        # global max pooling
        x = self.globalmaxpool(x)  # shape: (batch, 1, 1, 128)
        # flattening
        x = x.reshape(x.shape[0], -1)
        # fc layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)

# Hyperparameters
in_channels = 1
num_classes = 10
num_epochs = 4

# Hyperparameter (to search)
batch_sizes = [32, 64, 256]
learning_rates = [1e-1, 1e-2, 1e-3, 1e-4]
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Training/ hyperparameter searching
for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        step = 1
        # Instantiate network
        model = CNN(in_channels, num_classes)
        model.to(device=device)  # move to GPU
        model.train()  # toggle network to training mode
        loss_fn = nn.CrossEntropyLoss()  # loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # Data loader
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        # Setup tensorboard writer
        writer = SummaryWriter(log_dir=f"runs/MNIST/Minibatch size {batch_size} LR {learning_rate}")
        # Visualise the model in TensorBoard
        # Visualise image batches
        images, _ = next(iter(train_loader))
        images = images.to(device=device)
        writer.add_graph(model=model, input_to_model=images.to(device=device))
        writer.close()

        for epoch in range(num_epochs):
            losses = []
            accuracies = []
            #  batch index      batch (-1, 1, 28, 28)
            for batch_idx, (data, label) in enumerate(train_loader):
                # move data to GPU (if available)
                data = data.to(device=device)
                label = label.to(device=device)
                # forward
                prediction = model(data)
                loss = loss_fn(prediction, label)
                losses.append(loss.item())
                # calculate gradients
                optimizer.zero_grad()
                loss.backward()
                # weight update
                optimizer.step()

                # running accuracy
                _, predicted_labels = prediction.max(dim=1)
                num_correct_labels = (predicted_labels == label).sum()
                running_accuracy = float(num_correct_labels)/float(data.shape[0])
                accuracies.append(running_accuracy)

                # plotting to TensorBoard
                flattened_image = data.reshape(data.shape[0], -1)  # flatten all images in current batch
                image_grid = torchvision.utils.make_grid(data)  # visualise current batch as matrix
                predicted_class_labels = [classes[label] for label in predicted_labels]  # get predicted labels

                writer.add_image("mnist_images", image_grid)  # visualise current batch
                writer.add_histogram("fc3", model.fc3.weight)  # visualise weights and their updates
                writer.add_histogram("conv7", model.conv7.weight)
                writer.add_scalar("Training loss", loss, global_step=step)  # plot training loss
                writer.add_scalar("Training Accuracy", running_accuracy, global_step=step)  # plot training accuracy

                if batch_idx == 230:  # visualise the model's perception about the data
                    writer.add_embedding(
                        mat=flattened_image,
                        metadata=predicted_class_labels,  # predicted value
                        label_img=data,  # true value
                        global_step=batch_idx
                    )
                step += 1
            # add the current hyperparameters after processing the current batch
            writer.add_hparams(
                {"lr": learning_rate, "batch_size": batch_size},
                {"accuracy": sum(accuracies)/len(accuracies), "loss": sum(losses)/len(losses)}
            )

