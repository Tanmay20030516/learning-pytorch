import torch  # use updated version of pytorch
import torch.nn as nn
import torch.optim as optim
import spacy
from utils import (
    translate_sentence,
    bleu,
    save_checkpoint,
    load_checkpoint
)
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k

print("Imports done...")

# data pre-processing
spacy_eng = spacy.load("en_core_web_sm")
spacy_ger = spacy.load("de_core_news_sm")


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(
    tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>"
)
english = Field(
    tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>"
)
train_data, val_data, test_data = Multi30k.splits(
    exts=(".en", ".de"), fields=(english, german)
)
german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

print("Data set up done...")


# model building
class PositionalEmbedding(nn.Module):
    """
    Model contains no recurrence and no convolution, so to let the model know regarding the order of sequence
    some positional encoding is to be done for each word relative to it's position in the sequence
    https://nlp.seas.harvard.edu/annotated-transformer/#positional-encoding
    """

    def __init__(self, embed_dim, dropout_prob, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.positional_embedding = self.make_pos_emb(max_len, embed_dim).to(DEVICE)

    @staticmethod
    def make_pos_emb(max_len, embed_dim):
        pos_emb = torch.zeros(max_len, embed_dim)  # shape: (max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)  # shape: (max_len, 1)
        # 10000.0 ** (torch.arange(0, embed_dim, step=2)/embed_dim)
        # torch.exp(torch.log(10000.0 ** (torch.arange(0, embed_dim, step=2)/embed_dim)))
        denominator = torch.exp(
            (torch.arange(0, embed_dim, step=2) / embed_dim) * torch.log(torch.Tensor([10000.0])).item()
        )
        pos_emb[:, 0::2] = torch.sin(position / denominator)  # even indices in embed_dim
        pos_emb[:, 1::2] = torch.cos(position / denominator)  # odd indices in embed_dim
        pos_emb = pos_emb.unsqueeze(0)  # shape: (1, max_len, embed_dim)

        return pos_emb

    def forward(self, x):
        # x.shape: (N, seq_len, embed_dim)
        x = x + self.positional_embedding[:, : x.size(1), :]
        return self.dropout(x)


class Transformer_model(nn.Module):
    def __init__(
            self, embedding_size, src_vocab_size, src_pad_idx,
            trg_vocab_size, num_heads, num_encoder_layers, num_decoder_layers,
            forward_expansion, dropout_prob, max_len, device
    ):
        super(Transformer_model, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.trg_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.src_pos_embedding = PositionalEmbedding(embedding_size, dropout_prob, max_len)
        self.trg_pos_embedding = PositionalEmbedding(embedding_size, dropout_prob, max_len)
        self.device = device
        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=embedding_size*forward_expansion,
            dropout=dropout_prob
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        """
        makes padding mask for source sentence
        :param src: source sentence (seq_len, N)
        :return: mask for source sentence
        """
        src_mask = torch.tensor((src.transpose(0, 1) == self.src_pad_idx))  # True where ever there is padding idx
        return src_mask.to(self.device)  # (N, seq_len); becoz PyTorch transformer expect src_mask of (N, seq_len) shape

    def forward(self, src, trg):
        # src_seq_len, N = src.shape
        trg_seq_len, N = trg.shape

        src_embeddings = self.dropout(
            self.src_embedding(src) + self.src_pos_embedding(self.src_embedding(src))
        )
        trg_embeddings = self.dropout(
            self.trg_embedding(trg) + self.trg_pos_embedding(self.trg_embedding(trg))
        )

        src_pad_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_len).to(self.device)

        out = self.transformer(
            src_embeddings,
            trg_embeddings,
            src_key_padding_mask=src_pad_mask,
            tgt_mask=trg_mask
        )
        out = self.fc_out(out)
        return out


print("Transformer set up done...")

# hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOAD_MODEL = False
SAVE_MODEL = False
NUM_EPOCHS = 2
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
# model parameters
SRC_VOCAB_SIZE = len(english.vocab)
TRG_VOCAB_SIZE = len(german.vocab)
EMBEDDING_SIZE = 512
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DROPOUT = 0.10
MAX_LEN = 100
FORWARD_EXPANSION = 4
SRC_PAD_IDX = german.vocab.stoi['<pad>']
PAD_IDX = german.vocab.stoi['<pad>']

# tensorboard writer
writer = SummaryWriter("runs/loss_plot")
step = 0

train_iterator, val_iterator, test_iterator = BucketIterator.splits(
    (train_data, val_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=DEVICE
)

model = Transformer_model(
    embedding_size=EMBEDDING_SIZE,
    src_vocab_size=SRC_VOCAB_SIZE,
    trg_vocab_size=TRG_VOCAB_SIZE,
    src_pad_idx=SRC_PAD_IDX,
    num_heads=NUM_HEADS,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    forward_expansion=FORWARD_EXPANSION,
    dropout_prob=DROPOUT,
    max_len=MAX_LEN,
    device=DEVICE,
)
model = model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

if LOAD_MODEL:
    load_checkpoint(torch.load('checkpoint.pth.tar'), model, optimizer)

src_seq = "A big brown fox jumps over a tree."

for epoch in range(NUM_EPOCHS):
    print(f"[Epoch{epoch}/{NUM_EPOCHS}]")
    if SAVE_MODEL:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
    model.eval()
    translated_seq = translate_sentence(
        model, src_seq, english, german, DEVICE, MAX_LEN
    )
    print(f"Translated example sentence: \n {translated_seq}")
    model.train()
    losses = []

    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda
        inp_data = batch.src.to(DEVICE)
        target = batch.trg.to(DEVICE)

        # Forward prop
        output = model(inp_data, target[:-1, :])

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshaping
        # Let's also remove the start token while we're at it
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = loss_fn(output, target)
        losses.append(loss.item())

        # backward prop
        loss.backward()
        # gradient clipping to avoid exploding gradient issues, makes sure grads are within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # gradient descent update
        optimizer.step()

        # plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

    mean_loss = sum(losses) / len(losses)
    lr_scheduler.step(mean_loss)

print("Training done...")

print("Testing performance on test data...")
# running on entire test data takes a while
score = bleu(test_data[1:100], model, english, german, DEVICE)
print(f"Bleu score {score * 100:.2f}")
