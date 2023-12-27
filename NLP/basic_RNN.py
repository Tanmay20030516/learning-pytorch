import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = 28  # num of timestamps
HIDDEN_SIZE = 256
NUM_LAYERS = 2
NUM_CLASSES = 10
SEQUENCE_LENGTH = 28  # length of sequence passed at each timestamp
LEARNING_RATE = 0.004
BATCH_SIZE = 32
NUM_EPOCHS = 4


class RNN_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length):
        super(RNN_model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers,
            x.size(dim=0),  # batch size
            self.hidden_size
        ).to(DEVICE)
        # forward prop RNN
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        # decode the last hidden state (last time stamp)
        out = self.fc(out)

        return out


class GRU_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length):
        super(GRU_model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(
            self.num_layers,
            x.size(dim=0),  # batch size
            self.hidden_size
        ).to(DEVICE)
        # forward prop GRU
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        # decode the last hidden state (last time stamp)
        out = self.fc(out)

        return out


class LSTM_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length):
        super(LSTM_model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # in LSTMs we have a long term memory called cell state, apart from hidden statees
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)

        # forward pass
        out, _ = self.lstm(
            x, (h0, c0)
        )  # has shape (batch_size, seq_length, hidden_size)
        out = out.reshape(out.shape[0], -1)

        out = self.fc(out)
        return out


# load dataset
train_dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# initialize model
model_rnn = RNN_model(
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    NUM_CLASSES,
    SEQUENCE_LENGTH
).to(DEVICE)

model_gru = GRU_model(
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    NUM_CLASSES,
    SEQUENCE_LENGTH
).to(DEVICE)

model_lstm = LSTM_model(
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    NUM_CLASSES,
    SEQUENCE_LENGTH
).to(DEVICE)

# loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer_rnn = optim.Adam(model_rnn.parameters(), lr=LEARNING_RATE)
optimizer_gru = optim.Adam(model_gru.parameters(), lr=LEARNING_RATE)
optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=LEARNING_RATE)


def train_model(model, loss_fn, optimizer, num_epochs, data_loader):
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(data_loader)):
            data = data.to(DEVICE).squeeze(dim=1)  # remove the channels dimension
            targets = targets.to(DEVICE)

            pred = model(data)
            loss = loss_fn(pred, targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    # set model to eval mode
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE).squeeze(1)
            y = y.to(DEVICE)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    # set model back to train mode
    model.train()
    return num_correct / num_samples


# check for simple rnn
print("RNN")
train_model(
    model_rnn,
    loss_fn,
    optimizer_rnn,
    NUM_EPOCHS,
    train_loader
)
print(f"Accuracy on training set: {check_accuracy(train_loader, model_rnn)*100:2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model_rnn)*100:.2f}")
print('\n')
# check for gru
print("GRU")
train_model(
    model_gru,
    loss_fn,
    optimizer_rnn,
    NUM_EPOCHS,
    train_loader
)
print(f"Accuracy on training set: {check_accuracy(train_loader, model_rnn)*100:2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model_rnn)*100:.2f}")
print('\n')
# check for lstm
print("LSTM")
train_model(
    model_lstm,
    loss_fn,
    optimizer_rnn,
    NUM_EPOCHS,
    train_loader
)
print(f"Accuracy on training set: {check_accuracy(train_loader, model_rnn)*100:2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model_rnn)*100:.2f}")
print('\n')

