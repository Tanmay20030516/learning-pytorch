import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):  # (conv 3x3, ReLU)x2 blocks
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, out_channels_list=None):
        if out_channels_list is None:
            out_channels_list = [64, 128, 256, 512]

        super(UNET, self).__init__()

        self.downs = nn.ModuleList()  # storing list of modules
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # UNET down sampling (4 blocks)
        for out_channel_size in out_channels_list:
            self.downs.append(DoubleConv(in_channels, out_channel_size))
            in_channels = out_channel_size  # update in_channels

        # Bottleneck layer (lowermost)
        self.bottleneck = DoubleConv(in_channels=out_channels_list[-1],
                                     out_channels=out_channels_list[-1]*2)

        # UNET up sampling (4 blocks)
        for out_channel_size in reversed(out_channels_list):
            self.ups.append(  # in_channels=out_channel_size*2 because of channels concat from skip connection
                nn.ConvTranspose2d(out_channel_size*2, out_channel_size, kernel_size=2, stride=2)
            )  # green arrows (up-conv 2x2)
            self.ups.append(DoubleConv(out_channel_size*2, out_channel_size))  # (conv 3x3, ReLU)x2

        # Final convolution (channel reduction conv)
        self.final_conv = nn.Conv2d(in_channels=out_channels_list[0],
                                    out_channels=out_channels,
                                    kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # creating red arrows (max pool 2x2) [down sampling]
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # bottleneck layer
        x = self.bottleneck(x)
        # we now need the values in reverse while going in up sample part
        skip_connections.reverse()

        # creating green arrows [up sampling]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]

            if x.shape != skip_connection.shape:  # if w, h not a multiple of 16, there is flooring of values
                x = TF.resize(x, size=skip_connection.shape[2:])  # resize just h and w of the feature map

            concatenation = torch.cat([skip_connection, x], dim=1)  # channel concatenation
            x = self.ups[i+1](concatenation)

        # channel reduction convolutions
        x = self.final_conv(x)
        return x


def main():
    x = torch.randn((32, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    print(x.shape, model(x).shape)
    assert model(x).shape == x.shape, "height and width don't match, check concatenations"


if __name__ == "__main__":
    main()
