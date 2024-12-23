import math
import torch
import torch.nn as nn


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', inplace=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class EASTHead(nn.Module):
    def __init__(self, in_channels, scope):
        super(EASTHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, 1)  # 32, 1, 1
        self.sigmoid1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(in_channels, 4, 1)  # 32, 4, 1
        self.sigmoid2 = nn.Sigmoid()
        self.conv3 = nn.Conv2d(in_channels, 1, 1)  # 32, 1, 1
        self.sigmoid3 = nn.Sigmoid()
        self.scope = scope

    def forward(self, x):
        score = self.sigmoid1(self.conv1(x))
        loc = self.sigmoid2(self.conv2(x)) * self.scope
        angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi / 2
        geo = torch.cat((loc, angle), 1)
        return score, geo
