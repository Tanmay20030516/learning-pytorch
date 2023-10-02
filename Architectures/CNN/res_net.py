# ResNet paper: https://arxiv.org/pdf/1512.03385
# below comments are with respect to ResNet50

# Imports
import torch
import torch.nn as nn
from torchsummary import summary


class ResBlock(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_down_sample=None, stride=1):
        super(ResBlock, self).__init__()
        self.stride = stride
        self.intermediate_channels = intermediate_channels
        self.in_channels = in_channels
        self.identity_down_sample = identity_down_sample  # identity map (connection/ layer)

        self.expansion = 4  # channels quadruple after each block

        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False  # since using batch normalization
        )
        self.batchnorm1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.batchnorm2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels*self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.batchnorm3 = nn.BatchNorm2d(intermediate_channels*self.expansion)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()  # copy the inputs to input layer

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)

        if self.identity_down_sample is not None:
            # we enter this block only for 1st block for each of block1, block2, block3 and block4
            # these are generally 1x1 convolutions to match-up the output channels
            identity = self.identity_down_sample(identity)

        x += identity  # since the channels are matched up, we add the output and input
        x = self.relu(x)
        return x


print("Residual block implemented...")


class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes, ResBlock, res_block_list: list):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,  # "same" convolutions
            bias=False
        )  # o/p = (64, 112, 112)
        self.batchnorm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # o/p = (64, 56, 56)

        # now we begin making our residual blocks
        self.block1 = self.make_block(  # i/p = (64, 56, 56)
            ResBlock, num_residual_blocks=res_block_list[0], intermediate_channels=64, stride=1
        )  # (64, 56, 56)->[(64, 56, 56)->(64, 56, 56)->(256, 56, 56)]--->[(64, 56, 56)->(64, 56, 56)->(256, 56, 56)]--> ... so on
        #           \--> identity mapping -> (256, 56, 56)------------/ \-------------------------------------------/
        #                                    (added to above main path)
        self.block2 = self.make_block(
            ResBlock, num_residual_blocks=res_block_list[1], intermediate_channels=128, stride=2
        )
        self.block3 = self.make_block(
            ResBlock, num_residual_blocks=res_block_list[2], intermediate_channels=256, stride=2
        )
        self.block4 = self.make_block(
            ResBlock, num_residual_blocks=res_block_list[3], intermediate_channels=512, stride=2
        )

        # below type of pooling that fixes the output size, and accordingly handles input
        self.averagepool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.averagepool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    # noinspection PyShadowingNames
    def make_block(self, ResBlock: object, num_residual_blocks: int, intermediate_channels: int, stride: int):
        """
        :param stride: number of strides
        :param intermediate_channels: intermediate channels in the network
        :param num_residual_blocks: number of residual blocks
        :type ResBlock: class object that creates a residual block
        """
        identity_down_sample = None
        blocks = []

        # stride != 1 -> size of input i.e. n_H, n_W changed; for block2, block3, block4
        # self.in_channels != intermediate_channels*4; for first block of each of block1, block2, block3, block4
        if stride != 1 or self.in_channels != intermediate_channels*4:
            identity_down_sample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=intermediate_channels*4,  # expansion of channels at end of each res-block
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels*4)
            )

        blocks.append(  # the layer that changes number of channels
            ResBlock(self.in_channels, intermediate_channels, identity_down_sample, stride)
        )  # the identity map has corrected the input dimensions to match (intermediate_channels*4, .., ..)
        # so now we can directly add this updated channel (x) to output of remaining res-blocks

        self.in_channels = intermediate_channels*4
        # channel expansion after first block, so need to change the in_channel value for remaining res-blocks

        for i in range(num_residual_blocks-1):
            blocks.append(
                ResBlock(self.in_channels, intermediate_channels)
            )  # mapping 256 (in) -> 64 (out, of next block),... 64*4

        return nn.Sequential(*blocks)


print("ResNet implemented...")
print("Moving towards testing...")


def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(ResBlock=ResBlock, res_block_list=[3, 4, 6, 3], in_channels=img_channel, num_classes=num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(ResBlock=ResBlock, res_block_list=[3, 4, 23, 3], in_channels=img_channel, num_classes=num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(ResBlock=ResBlock, res_block_list=[3, 8, 36, 3], in_channels=img_channel, num_classes=num_classes)


def main():
    BATCH_SIZE = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet101().to(device)
    output = model((torch.randn(BATCH_SIZE, 3, 224, 224)).to(device))
    assert output.size() == torch.Size([BATCH_SIZE, 1000])
    print(output.shape)
    print(summary(model, (3, 224, 224)))


if __name__ == "__main__":
    main()
