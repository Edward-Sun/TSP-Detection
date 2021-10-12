import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.utils import env
from detectron2.layers.batch_norm import NaiveSyncBatchNorm

class MyConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, kernel_size=3, stride=1):
        super().__init__()
        out_channels = out_channels or in_channels
        if kernel_size == 3:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        elif kernel_size == 1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        else:
            raise NotImplementedError
        self.norm = norm
        if self.norm:
            SyncBN = NaiveSyncBatchNorm if env.TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm
            self.bn = SyncBN(num_features=out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.bn(x)
        if self.activation:
            x = F.relu(x, inplace=True)
        return x

class MySepConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False):
        super().__init__()
        out_channels = out_channels or in_channels
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.norm = norm
        if self.norm:
            SyncBN = NaiveSyncBatchNorm if env.TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm
            self.bn = SyncBN(num_features=out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        if self.norm:
            x = self.bn(x)
        if self.activation:
            x = F.relu(x, inplace=True)
        return x
