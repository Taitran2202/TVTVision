import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from .module import Extractor


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class ReduceCh(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(ReduceCh, self).__init__()
        assert (len(in_chs) == len(out_chs) == 4)
        self.conv1 = nn.Conv2d(in_chs[0], out_chs[0], kernel_size=1)
        self.conv2 = nn.Conv2d(in_chs[1], out_chs[1], kernel_size=1)
        self.conv3 = nn.Conv2d(in_chs[2], out_chs[2], kernel_size=1)
        self.conv4 = nn.Conv2d(in_chs[3], out_chs[3], kernel_size=1)

    def forward(self, c2, c3, c4, c5):
        c5 = self.conv1(c5)
        c4 = self.conv2(c4)
        c3 = self.conv3(c3)
        c2 = self.conv4(c2)
        return c2, c3, c4, c5


class Head(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Head, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1,
                      bias=False), nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1,
                      bias=False), nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1,
                      bias=False), nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1))

    def forward(self, x):
        return self.conv(x)


class DCLNet(nn.Module):
    def __init__(self, pretrained=False, backbone="resnet50"):
        super(DCLNet, self).__init__()
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.backbone = Extractor(pretrained, backbone)
        in_chs = [2048, 1024, 512, 256] if backbone == 'resnet50' else [
            512, 256, 128, 64]
        out_chs = [i//2 for i in in_chs]
        self.redu_cls = ReduceCh(in_chs, out_chs)

        self.upconv1 = DoubleConv(out_chs[0] + out_chs[1], out_chs[1])
        self.upconv2 = DoubleConv(out_chs[1] + out_chs[2], out_chs[2])
        self.upconv3 = DoubleConv(out_chs[2] + out_chs[3], out_chs[3])

        self.head_cls = Head(out_chs[-1], 1)
        self.head_rho = Head(out_chs[-1], 4)
        self.head_theta = Head(out_chs[-1], 4)

    def forward(self, x):
        x = self.normalize(x)
        c2, c3, c4, c5 = self.backbone(x)

        c2, c3, c4, c5 = self.redu_cls(c2, c3, c4, c5)
        y = F.interpolate(c5, size=c4.size()[
                          2:], mode='bilinear', align_corners=True)
        y = torch.cat([y, c4], dim=1)
        y1 = self.upconv1(y)

        y = F.interpolate(y1, size=c3.size()[
                          2:], mode='bilinear', align_corners=True)
        y = torch.cat([y, c3], dim=1)
        y2 = self.upconv2(y)

        y = F.interpolate(y2, size=c2.size()[
                          2:], mode='bilinear', align_corners=True)
        y = torch.cat([y, c2], dim=1)
        fuse = self.upconv3(y)

        cls = torch.sigmoid(self.head_cls(fuse))
        rho = F.relu(self.head_rho(fuse), inplace=True)
        theta = torch.sigmoid(self.head_theta(fuse)) * 2 * math.pi
        return cls, rho, theta
