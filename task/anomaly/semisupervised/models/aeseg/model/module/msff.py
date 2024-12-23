import torch.nn as nn
import torch.nn.functional as F
from .coordatt import CoordAtt
from .se import SE


class MSFFBlock(nn.Module):
    def __init__(self, in_channel, use_se=True):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True)
        )
        self.attn = SE(in_channel, ratio=4) if use_se else CoordAtt(
            in_channel, in_channel)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 2, 3, padding=1),
            nn.BatchNorm2d(in_channel // 2),
            nn.ReLU(True),
            nn.Conv2d(in_channel // 2, in_channel // 2, 3, padding=1),
            nn.BatchNorm2d(in_channel // 2),
            nn.ReLU(True)
        )

    def forward(self, x):
        x_conv = self.conv1(x)
        x_att = self.attn(x)
        x = x_conv * x_att
        x = self.conv2(x)
        return x


class MSFF(nn.Module):
    def __init__(self, use_se: bool = True):
        super().__init__()
        self.blk1 = MSFFBlock(128, use_se)
        self.blk2 = MSFFBlock(256, use_se)
        self.blk3 = MSFFBlock(512, use_se)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.upconv32 = nn.Sequential(
            self.up,
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.upconv21 = nn.Sequential(
            self.up,
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        for m in [self.blk1, self.blk2, self.blk3, self.upconv32, self.upconv21]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, features):
        # features = [level1, level2, level3]
        f1, f2, f3 = features

        # MSFF Module
        f1_k = self.blk1(f1)
        f2_k = self.blk2(f2)
        f3_k = self.blk3(f3)

        f2_f = f2_k + self.upconv32(f3_k)
        f1_f = f1_k + self.upconv21(f2_f)

        # spatial attention
        # mask
        # m3 = f3[:, 256:, ...].mean(dim=1, keepdim=True)
        # m2 = f2[:, 128:, ...].mean(dim=1, keepdim=True) * self.up(m3)
        # m1 = f1[:, 64:, ...].mean(dim=1, keepdim=True) * self.up(m2)

        m3 = f3[:, 256:, ...].mean(dim=1, keepdim=True)
        m2 = f2[:, 128:, ...].mean(dim=1, keepdim=True) * F.interpolate(
            m3, scale_factor=2, mode='bilinear', align_corners=True)
        m1 = f1[:, 64:, ...].mean(dim=1, keepdim=True) * F.interpolate(
            m2, scale_factor=2, mode='bilinear', align_corners=True)

        # m3_temp = f3[:, 256:, ...].mean(dim=1, keepdim=True)
        # m2_temp = f2[:, 128:, ...].mean(dim=1, keepdim=True)
        # m1_temp = f1[:, 64:, ...].mean(dim=1, keepdim=True)

        # m3 = F.interpolate(m3_temp, size=(64, 64), mode='bilinear', align_corners=False)
        # m2 = F.interpolate(m2_temp, size=(64, 64), mode='bilinear', align_corners=False)
        # m1 = m1_temp * m2 * m3

        # m3 = F.interpolate(m3_temp, size=(32, 32), mode='bilinear', align_corners=False)
        # m2 = m2_temp * m3

        # m3 = m3_temp

        f1_out = f1_f * m1
        f2_out = f2_f * m2
        f3_out = f3_k * m3

        return [f1_out, f2_out, f3_out]
