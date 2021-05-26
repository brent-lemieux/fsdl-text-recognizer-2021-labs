from typing import Any, Dict
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


CONV_DIM = 64
FC_DIM = 128
IMAGE_SIZE = 28
STRIDE = 2
DROPOUT = .1


class ConvBlock(nn.Module):
    """
    ResNet BasicBlock.
    """
    def __init__(self, input_channels: int, output_channels: int,
                 kernel_size: int = 3, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=kernel_size, stride=1, dilation=1, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

        self.expansion = 1
        if stride > 1:
          self.downsample = nn.Sequential(
              nn.Conv2d(input_channels, output_channels * self.expansion, kernel_size=1, stride=stride),
              nn.BatchNorm2d(output_channels * self.expansion),
          )
        else:
          self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            of dimensions (B, C, H, W)

        Returns
        -------
        torch.Tensor
            of dimensions (B, C, H, W)
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample != None:
          x = self.downsample(x)
        out += x
        out = self.relu(out)
        return out


class CNN(nn.Module):
    """Simple CNN for recognizing characters in a square image."""

    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        input_dims = data_config["input_dims"]
        num_classes = len(data_config["mapping"])

        conv_dim = self.args.get("conv_dim", CONV_DIM)
        fc_dim = self.args.get("fc_dim", FC_DIM)
        dropout = self.args.get("dropout", DROPOUT)
        stride = self.args.get("stride", STRIDE)

        self.conv1 = ConvBlock(input_dims[0], conv_dim, stride=stride)
        self.conv2 = ConvBlock(conv_dim, conv_dim, stride=stride)
        self.dropout = nn.Dropout(dropout)
        # self.max_pool = nn.MaxPool2d(2)

        # Because our 3x3 convs have padding size 1, they leave the input size unchanged.
        # The 2x2 max-pool divides the input size by 2. Flattening squares it.
        conv_output_size = IMAGE_SIZE // 2
        fc_input_dim = int(conv_output_size * conv_output_size * conv_dim)
        self.fc1 = nn.Linear(fc_input_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x
            (B, C, H, W) tensor, where H and W must equal IMAGE_SIZE

        Returns
        -------
        torch.Tensor
            (B, C) tensor
        """
        _B, _C, H, W = x.shape
        assert H == W == IMAGE_SIZE
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.max_pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--conv_dim", type=int, default=CONV_DIM)
        parser.add_argument("--fc_dim", type=int, default=FC_DIM)
        return parser
