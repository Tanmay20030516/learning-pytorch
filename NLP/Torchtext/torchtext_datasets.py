import spacy
import torch
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

spacy_en = spacy.load("en_core_web_sm")
spacy_gr = spacy.load("de_core_news_sm")


def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]
# print(next(token.text for token in spacy_en.tokenizer("text hai ye bhayy")))


def tokenize_gr(text):
    return [token.text for token in spacy_gr.tokenizer(text)]


english = Field(sequential=True, use_vocab=True, tokenize=tokenize_en, lower=True)  # contains instructions for conversion to tensor
german = Field(sequential=True, use_vocab=True, tokenize=tokenize_gr, lower=True)


train_data, validation_data, test_data = Multi30k.splits(
    exts=(".de", ".en"),  # german to english
    fields=(german, english)
)
# create vocabulary from train_data
english.build_vocab(train_data, max_size=10000, min_freq=2)
german.build_vocab(train_data, max_size=10000, min_freq=2)

train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    datasets=(train_data, validation_data, test_data),
    batch_size=64,
    device=torch.device("cuda") if torch.cuda.is_available() else "cpu"
)


for batch in train_iterator:
    print(batch)

# string to integer (stoi ~ str_to_idx)
print(f'Index of the word "the" is: {english.vocab.stoi["the"]}')
print(f'Index of the word "cat" is: {english.vocab.stoi["cat"]}')

# print integer to string (itos ~ idx_to_str)
print(f"Word at the index 1612 is: {english.vocab.itos[1612]}")
print(f"Word at the index 0 is: {english.vocab.itos[0]}")
