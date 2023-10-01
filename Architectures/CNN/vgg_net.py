# VGG paper: https://arxiv.org/pdf/1409.1556.pdf

# Imports
import torch
import torch.nn as nn
from torchsummary import summary

# VGG Net variations
VGGNets = {  # just the conv layer section, fully connected layers were the same for all variations
    # number => out-channels of conv layer
    # letter => max-pooling layer
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


# Implementation
class VGGNet(nn.Module):
    def __init__(self, in_channels, num_classes, variant):
        super(VGGNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv = self.create_conv_layers(VGGNets[variant])
        self.fcs = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        network = []
        for layer_idx, num_out_channels in enumerate(architecture):
            in_channels = architecture[layer_idx - 1]
            if type(num_out_channels) == int:

                if layer_idx == 0:
                    in_channels = 3
                elif architecture[layer_idx - 1] == 'M':
                    in_channels = architecture[layer_idx - 2]

                out_channels = num_out_channels
                network.append(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
                # batch-norm not in actual paper
                network.append(nn.BatchNorm2d(out_channels))  # batch-norm applied at dim=1 (across channels)
                network.append(nn.ReLU())
            elif num_out_channels == 'M':
                network.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*network)  # unpack the elements in list, and pass as individual element to nn.Sequential


# Summarize current architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGGNet(3, 1000, "VGG11").to(device)
print(summary(model, (3, 224, 224)))

# Checking
BATCH_SIZE = 8
data = torch.randn(BATCH_SIZE, 3, 224, 224).to(device)
assert model(data).shape == torch.Size([BATCH_SIZE, 1000]), "Re-check model architecture"
print(model(data).shape)
