import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class SE(nn.Module):
    def __init__(self, in_chnls, ratio=7, out_chnls=66):
        super(SE, self).__init__()
        self.f = BasicConv2d(in_chnls, in_chnls, kernel_size=3, padding=1)
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        # b, c, h, w = x.size()
        # out = self.f(x)
        out = self.squeeze(x)
        # out = out.view(b,c)
        out = self.compress(out)
        out = F.relu(out)
        out_t = self.excitation(out)  # .view(b,c,1,1)
        out = torch.sigmoid(out_t)
        return out
