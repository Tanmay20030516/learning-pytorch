# YOLOv1 paper: https://arxiv.org/pdf/1506.02640.pdf
# image split into 7x7 grid, and each cell has 2 types of bboxes (kinda anchor boxes)
# each cell can output only one object detections

import torch
import torch.nn as nn

architecture_config = [  # convolutional part
    (7, 64, 2, 3),  # tuple: (kernel_size, out_channels, stride, padding)
    "M",  # str (max pooling): (kernel_size, stride)
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    # list: [tuple1, tuple2, ..., number_of_sequential_repeats_of_tuples]
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=False,
            **kwargs,  # kernel_size, stride, ...
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)  # not originally
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.batchnorm(self.conv(x))
        return self.leakyrelu(x)


class YOLOv1(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(YOLOv1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self.create_conv_layers(self.architecture)
        self.fc = self.create_fully_connected_layers(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        # alternate for .reshape(); dim=0 is batches (examples), taken care of by nn.Flatten() later on
        # x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for comp in architecture:
            if type(comp) == tuple:
                layers += [
                    CNNBlock(
                        in_channels=in_channels,
                        out_channels=comp[1],
                        kernel_size=comp[0],
                        stride=comp[2],
                        padding=comp[3],
                    )
                ]
                in_channels = comp[1]  # update in channels for next block

            elif type(comp) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            elif type(comp) == list:
                # currently we have two conv layers for sequential repetition
                conv1 = comp[0]
                conv2 = comp[1]
                num_repeats = comp[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels=in_channels,
                            out_channels=conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            in_channels=conv1[1],
                            out_channels=conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]  # update in channels for next set of repetitions

        return nn.Sequential(*layers)

    def create_fully_connected_layers(self, split_size, num_boxes_per_cell, num_classes):
        S, B, C = split_size, num_boxes_per_cell, num_classes
        final_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024 * S * S, out_features=4096),
            # nn.Dropout(0.0),  # not originally
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=4096, out_features=S * S * (C + B * 5)),  # target of shape S*S*(C+5)
                                                                            # needs to reshaped as (S, S, C + B*5), taking care of batches dimension
        )
        return final_block


def _try_(S=7, B=2, C=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # split_size, num_boxes_per_cell, num_classes become keyword arguments
    model = YOLOv1(in_channels=3, split_size=S, num_boxes_per_cell=B, num_classes=C).to(device=device)
    x = torch.randn((1, 3, 448, 448)).to(device=device)
    print(model(x).shape)  # o/p -> torch.Size([1, 1470])


if __name__ == "__main__":
    _try_()
