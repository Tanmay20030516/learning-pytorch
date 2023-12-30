import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from torchtext.data import Field, TabularDataset, BucketIterator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
spacy_en = spacy.load("en_core_web_sm")


def tokenize(text):
    return [token.text for token in spacy_en.tokenizer(text)]


# quote column in dataset
quote = Field(
    sequential=True,  # if data is sequential in nature
    use_vocab=True,
    tokenize=tokenize,
    lower=True,
)
# the score of the quote (positive - 1; negative - 0)
score = Field(
    sequential=False,
    use_vocab=False,
)

fields = {
    "quote": ("q", quote),  # "q" is the way to extract quote from current batch i.e. batch.q gives quotes
    "score": ("s", score),  # similarly for score
}

train_data, test_data = TabularDataset.splits(
    path="data",  # folder containing data
    train="train.json",  # file name
    test="test.json",
    format="json",  # csv, tsv, json
    fields=fields,  # fields created
)

print("dataset created...")

quote.build_vocab(
    train_data,
    max_size=10000,
    min_freq=1,
    # vectors="glove.6B.100d",  # express each word as word embeddings (glove vectors)
)

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data), batch_size=2, device=device
)

print("data iterators set...")

def visualize_data(iterator):
    for batch_idx, batch in enumerate(iterator):
        # get data to cuda if possible
        data = batch.q.to(device=device)
        print("data of shape (max_seq_len_in_curr_batch, batch_size)")
        print(data.numpy())
        # visualise the string form of train data
        for i in range(data.numpy().shape[-1]):
            str_data = []
            for idx in data.numpy():
                str_data.append(quote.vocab.itos[idx[i]])
            print(str_data)


# visualize_data(train_iterator)


# training a network on above dataset (this is a sentiment analysis problem)
class LSTM_model(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers):
        super(LSTM_model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers)
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(device)

        embedded = self.embedding(x)
        outputs, _ = self.rnn(embedded, (h0, c0))
        prediction = self.fc_out(outputs[-1, :, :])

        return prediction


# Hyperparameters
input_size = len(quote.vocab)
hidden_size = 512
num_layers = 2
embedding_size = 100
learning_rate = 0.005
num_epochs = 10

# Initialize network
model = LSTM_model(input_size, embedding_size, hidden_size, num_layers).to(device)

print("model initialized...")

# load the pretrained embeddings (glove vectors) onto our model [uncomment vectors arg while creating vocabulary above]
# pretrained_embeddings = quote.vocab.vectors
# model.embedding.weight.data.copy_(pretrained_embeddings)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("optimiser and loss function set...")

# Train Network
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_iterator):
        # get data to cuda if possible
        data = batch.q.to(device=device)
        targets = batch.s.to(device=device)
        # forward pass
        scores = model(data)
        loss = criterion(scores.squeeze(1), targets.type_as(scores))
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # weight update
        optimizer.step()


print("training done...")