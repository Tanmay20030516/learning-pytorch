# InceptionNet (GoogleNet) paper: https://arxiv.org/pdf/1409.4842.pdf

# Imports
import torch
import torch.nn as nn
from torchsummary import summary

# To implement: conv_block, inception_block, auxiliary_blocks, inception_net (google_net)
#                                                    |
#                                           (these blocks are created so that we can check while training
#                                            if the network trained till current block over-fits or under-fits)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


print("conv block created...")


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1_pool):
        super(InceptionBlock, self).__init__()
        self.branch1 = ConvBlock(  # 1x1 block
            in_channels=in_channels, out_channels=out_1x1, kernel_size=1
        )
        self.branch2 = nn.Sequential(  # 3x3 block
            ConvBlock(in_channels=in_channels, out_channels=red_3x3, kernel_size=1),  # 1x1 channel reduction convolutions
            ConvBlock(in_channels=red_3x3, out_channels=out_3x3, kernel_size=3, padding=1),  # "same" padding
        )
        self.branch3 = nn.Sequential(  # 5x5 block
            ConvBlock(in_channels=in_channels, out_channels=red_5x5, kernel_size=1),  # 1x1 channel reduction convolutions
            ConvBlock(in_channels=red_5x5, out_channels=out_5x5, kernel_size=5, padding=2),  # "same" padding
        )
        self.branch4 = nn.Sequential(  # pooling block
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # 3x3 "same" pooling
            ConvBlock(in_channels=in_channels, out_channels=out_1x1_pool, kernel_size=1),  # 1x1 conv block
        )

    def forward(self, x):
        x = torch.cat(  # channel (along dim=1) concatenation
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1
        )
        return x


print("inception block created...")


class InceptionAux(nn.Module):
    """
    these blocks added after inception block 4a and 4d (only while training)
    """
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)
        self.averagepool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.averagepool(x)
        x = self.conv(x)

        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


print("auxiliary block created...")


class InceptionNet(nn.Module):
    def __init__(self, aux_logits=True, num_classes=1000):
        super(InceptionNet, self).__init__()
        assert aux_logits == True or aux_logits == False  # any value for aux_logits other than True/False raises an error
        self.aux_logits = aux_logits
        # the network components
        # input image - (3, 224, 224)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)  # o/p -> (64, 112, 112)

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # o/p -> (64, 56, 56)
        self.conv2 = nn.Conv2d(64, 192, 3, 1, 1)  # o/p -> (192, 56, 56)
        self.maxpool2 = nn.MaxPool2d(3, 2, 1)  # o/p -> (192, 28, 28) = (192, floor(56-3+2(1)/2 + 1), floor(56-3+2(1)/2 + 1))

        # after channel concatenation (happens after each inception block) => in_channels = out_1x1 + out_3x3 + out5x5 + out_1x1_pool
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)  # o/p -> (256, 28, 28)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)  # o/p -> (480, 28, 28)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # o/p -> (480, 14, 14)

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)  # o/p -> (512, 14, 14)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)  # o/p -> (512, 14, 14)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)  # o/p -> (512, 14, 14)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)  # o/p -> (528, 14, 14)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)  # o/p -> (832, 14, 14)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # o/p -> (832, 7, 7)

        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)  # o/p -> (832, 7, 7)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)  # o/p -> (1024, 7, 7)

        # like global average pooling layer
        self.averagepool = nn.AvgPool2d(kernel_size=7, stride=1)  # o/p -> (1024, 1, 1)
        self.dropout = nn.Dropout(p=0.4)  # o/p -> (1024, 1, 1)
        self.fc1 = nn.Linear(1024, num_classes)  # o/p -> (1000, 1, 1)

        # auxiliary blocks (used during training only)
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)  # outputs of 4a block fed
            self.aux2 = InceptionAux(528, num_classes)  # outputs of 4d block fed
        else:
            self.aux1 = self.aux2 = None

    def forward(self, x):  # refer pg 6-8 of research paper
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)

        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.averagepool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)

        if self.aux_logits and self.training:
            return aux1, aux2, x
        else:
            return x


print("inceptionNet (googleNet) model created...")

print("testing implemented model...")
if __name__ == "__main__":
    BATCH_SIZE = 4
    x = torch.randn(BATCH_SIZE, 3, 224, 224).to(torch.device("cuda"))
    model = InceptionNet(aux_logits=True, num_classes=1000).to(torch.device("cuda"))
    print(summary(model, (3, 224, 224)))
    print(model(x)[2].shape)
    assert model(x)[2].shape == torch.Size([BATCH_SIZE, 1000])


