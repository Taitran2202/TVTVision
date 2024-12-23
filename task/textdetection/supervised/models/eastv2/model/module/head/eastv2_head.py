import torch.nn as nn
import torch.nn.functional as F


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', inplace=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class EASTHead(nn.Module):
    def __init__(self, in_channels, scope):
        super(EASTHead, self).__init__()
        self.det_conv1 = ConvBN(in_channels, in_channels//2, 3, 1, 1)
        self.det_conv2 = ConvBN(in_channels//2, in_channels//4, 3, 1, 1)
        self.score_conv = nn.Conv2d(in_channels//4, 1, 1, 1)
        self.geo_conv = nn.Conv2d(in_channels//4, 8, 1, 1)
        self.scope = scope

    def forward(self, x):
        f_det = self.det_conv1(x)
        f_det = self.det_conv2(f_det)
        f_score = self.score_conv(f_det)
        f_score = F.sigmoid(f_score)
        f_geo = self.geo_conv(f_det)
        f_geo = (F.sigmoid(f_geo) - 0.5) * 2 * 800

        return f_score, f_geo
