# torchtext version == 0.4

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

spacy_eng = spacy.load("en_core_web_sm")
spacy_ger = spacy.load("de_core_news_sm")


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(
    init_token="<sos>",
    eos_token="<eos>",
    lower=True,
    tokenize=tokenize_ger,
)
english = Field(
    init_token="<sos>",
    eos_token="<eos>",
    lower=True,
    tokenize=tokenize_eng,
)

# train data loading
train_data, validation_data, test_data = Multi30k.splits(exts=(".de", ".en"), fields=(german, english))

# build vocabulary
german.build_vocab(train_data, min_freq=2, max_size=10000)
english.build_vocab(train_data, min_freq=2, max_size=10000)
# print(f"length of german vocab {len(german.vocab)}")
# print(f"length of english vocab{len(english.vocab)}")


# creating model
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout_prob):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)  # input_size is vocab size
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_prob)

    def forward(self, x):
        # x.shape = (seq_len, N) , where seq_len is no. of time steps and N is batch_size
        embeddings = self.dropout(self.embedding(x))
        # embeddings.shape = (seq_len, N, embedding_size)
        _, (hidden_states, cell_states) = self.rnn(embeddings)
        # hidden_states.shape = (seq_length, N, hidden_size)
        # cell_states.shape = (seq_length, N, hidden_size)

        return hidden_states, cell_states


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout_prob):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state, cell_state):
        x = x.unsqueeze(dim=0)  # reshaping x to (1, N) since we send/predict one word at a time (seq_length=1)
        embeddings = self.dropout(self.embedding(x))
        # embeddings - (1, N, embedding_size)
        outputs, (hidden_state, cell_state) = self.rnn(embeddings, (hidden_state, cell_state))
        # outputs - (1, N, hidden_size)
        # hidden_state, cell_state - (1, N, hidden_size)
        predictions = self.fc(outputs)
        # predictions - (1, N, len(target_vocabulary))
        predictions = predictions.squeeze(dim=0)  # remove the extra dim
        # now it becomes, predictions - (N, len(target_vocabulary))
        return predictions, (hidden_state, cell_state)


class Seq2SeqModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]  # source - torch.Size([17, 64]) (as an example)
        # print("source.shape[1] = ", source.shape)
        target_len = target.shape[0]  # target - torch.Size([22, 64]) (as an example)
        # print("target.shape[0] = ", target.shape)
        target_vocab_size = len(english.vocab)
        # print("target_vocab_size = ", target_vocab_size)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(DEVICE)
        # here, for 1st batch example (1st column),
        # each row (target_len num of rows) has a target_vocab_size length dim (along z direction),
        # which carries the index of probable word predicted by decoder

        hidden_state, cell_state = self.encoder(source)

        x = target[0]  # first input to decoder is "<sos>" token

        for t in range(1, target_len):
            # using previous hidden and cell states as context
            output, (hidden_state, cell_state) = self.decoder(x, hidden_state, cell_state)
            # print("output, (hidden_state, cell_state) = self.decoder(x, hidden_state, cell_state)")
            # print(output.shape, " - ", hidden_state.shape, " - ", cell_state.shape)
            # (batch_size, target_vocab_size) - (2, batch_size, embedding_size) - (2, batch_size, embedding_size)
            # store the next output prediction
            outputs[t] = output  # output.shape - (N, target_vocab_size)
            # get index of the best word (most probable) predicted by decoder
            # this index can be used to find word (idx_to_str)
            best_guess = output.argmax(axis=1)

            # the ratio uses the correct word as a word to be passed to current step rather than previous word predicted
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs


print("Model setup done...")

# Network training
print("Starting model training...")
NUM_EPOCHS = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 64
# Model hyperparameters
LOAD_MODEL = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE_ENCODER = len(german.vocab)
INPUT_SIZE_DECODER = len(english.vocab)
OUTPUT_SIZE = len(english.vocab)
ENCODER_EMBEDDING_SIZE = 300
DECODER_EMBEDDING_SIZE = 300
HIDDEN_SIZE = 1024  # needs to be the same for both encoder and decoder
NUM_LAYERS = 2
ENCODER_DROPOUT = 0.5
DECODER_DROPOUT = 0.5

writer = SummaryWriter(f"runs/loss_plot")
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    datasets=(train_data, validation_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=DEVICE,
)

encoder = Encoder(
    input_size=INPUT_SIZE_ENCODER,
    embedding_size=ENCODER_EMBEDDING_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout_prob=ENCODER_DROPOUT,
)
encoder = encoder.to(DEVICE)

decoder = Decoder(
    input_size=INPUT_SIZE_DECODER,
    embedding_size=DECODER_EMBEDDING_SIZE,
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_SIZE,
    num_layers=NUM_LAYERS,
    dropout_prob=DECODER_DROPOUT,
)
decoder = decoder.to(DEVICE)

pad_idx = english.vocab.stoi["<pad>"]

MT_model = Seq2SeqModel(encoder, decoder).to(DEVICE)
optimizer = optim.Adam(MT_model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

if LOAD_MODEL:
    load_checkpoint(torch.load("checkpoint.pth.tar"), MT_model, optimizer)

sample_sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."

for epoch in range(NUM_EPOCHS):
    print(f"[Epoch {epoch+1} / {NUM_EPOCHS}]")
    checkpoint = {"state_dict": MT_model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint)

    MT_model.eval()

    translated_sentence = translate_sentence(
        MT_model, sample_sentence, german, english, DEVICE, max_length=50
    )

    print(f"Translated example sentence: \n {translated_sentence}")

    MT_model.train()

    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(DEVICE)
        target = batch.trg.to(DEVICE)

        # forward prop
        output = MT_model(inp_data, target)  # shape = (target_len, batch_size, output_dim)

        # output is of shape (target_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshaping.
        # while we're at it let's also remove the start token.
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = loss_fn(output, target)

        # back prop
        loss.backward()

        # gradient clipping to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(MT_model.parameters(), max_norm=1)

        # gradient descent step
        optimizer.step()

        # plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1


print("Training done...")
print("Testing BLEU score...")
# let us see bleu score
score = bleu(test_data[1:100], MT_model, german, english, DEVICE)
print(f"Bleu score {score*100:.4f}")


