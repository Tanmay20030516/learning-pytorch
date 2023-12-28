import torch
import torch.nn as nn
import torchvision.models as models
import statistics
from torchsummary import summary
from torchvision.models import ResNet50_Weights

# print(models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1))


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, images):
        features = self.resnet(images)
        # print(features.shape)  # (batch_size, embed_size)
        # for name, param in self.resnet.named_parameters():
        #     if "fc.weight" in name or "fc.bias" in name:
        #         param.requires_grad = True  # unfreeze the last layer
        #     else:
        #         param.requires_grad = self.train_CNN  # freeze the rest layers

        return self.dropout(self.relu(features))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)  # take output from lstm to map it to vocab
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, features, captions):
        """
        forward pass for decoder RNN
        :param features: output from CNN
        :param captions: target captions
        :return:
        """
        embeddings = self.dropout(self.embed_layer(captions))
        embeddings = torch.cat(
            # features.unsqueeze(dim=0) adds dimension for time stamp;
            # features used for the first word prediction by LSTM
            tensors=(features.unsqueeze(dim=0), embeddings),
            dim=0
        )
        hidden_states, _ = self.lstm(embeddings)
        outputs = self.linear(hidden_states)

        return outputs


class CaptionModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CaptionModel, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    # for inference
    def caption_image(self, image, vocabulary, max_length=50):
        """
        used during inference
        :param image:
        :param vocabulary:
        :param max_length:
        :return:
        """
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(dim=0)
            states = None

            for _ in range(max_length):
                hidden_states, cell_states = self.decoderRNN.lstm(x, states)
                # linear layer maps hidden states to vocabulary words (softmax function)
                output = self.decoderRNN.linear(hidden_states.squeeze(dim=0))  # remove the added time step dimension, to make suitable for linear layer
                pred_word_idx = output.argmax(axis=1)
                result_caption.append(pred_word_idx.item())
                # use the previously predicted word to predict next word
                x = self.decoderRNN.embed_layer(pred_word_idx).unsqueeze(dim=0)
                if vocabulary.idx_to_str[pred_word_idx.item()] == "<EOS>":
                    break  # stop prediction as soon as <EOS> token is generated

        return [vocabulary.idx_to_str[idx] for idx in result_caption]
