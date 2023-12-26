# https://www.kaggle.com/datasets/adityajn105/flickr8k/data

import os

import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# python -m spacy download en_core_web_sm
spacy_eng = spacy.load("en_core_web_sm")
# print(spacy_eng)

class Vocabulary:
    """
    create vocabulary for text corpus
    """
    def __init__(self, freq_threshold):
        self.idx_to_str = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.str_to_idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.idx_to_str)

    @staticmethod  # bounding this method only to this class
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        freqs = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in freqs:
                    freqs[word] = 1
                else:
                    freqs[word] += 1

                if freqs[word] == self.freq_threshold:
                    self.str_to_idx[word] = idx
                    self.idx_to_str[idx] = word
                    idx = idx + 1

    def index_encoding(self, text):
        tokenized_text = self.tokenizer_eng(text)

        # list comprehension
        return [
            self.str_to_idx[token] if token in self.str_to_idx else self.str_to_idx["<UNK>"] for token in tokenized_text
        ]


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        # access dataframe columns
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        # initialize and build the vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        getting a single example out of the dataset
        :param idx: the index of example to access
        :return: a single example
        """
        caption = self.captions[idx]
        img_id = self.imgs[idx]

        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        numeric_encoded_caption = [self.vocab.str_to_idx["<SOS>"]]
        numeric_encoded_caption += self.vocab.index_encoding(caption)
        numeric_encoded_caption += self.vocab.str_to_idx["<EOS>"]

        return img, torch.tensor(numeric_encoded_caption)


class MyCollate:
    """
    pads sentences if needed for uniform sentence length across a batch (rather than across full corpus)
    """
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [pair[0].unsqueeze(0) for pair in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [pair[1] for pair in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


def get_loader(
        root_folder, caption_file, transform, batch_size=32, num_workers=8, shuffle=True, pin_memory=True
):
    """
    get the data loader for text file
    :param root_folder:
    :param caption_file:
    :param transform:
    :param batch_size:
    :param num_workers:
    :param shuffle:
    :param pin_memory:
    :return:
    """
    data = FlickrDataset(root_folder, caption_file, transform)

    pad_idx = data.vocab.str_to_idx["<PAD>"]

    loader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx)
    )

    return loader, data


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    loader, dataset = get_loader(
        "flickr8k/images/", "flickr8k/captions.txt", transform=transform
    )

    for idx, (imgs, captions) in enumerate(loader):
        print(imgs.shape)
        print(captions.shape)

