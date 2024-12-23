import timm
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResidualBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, inputs):
        x1 = self.conv(inputs)
        x2 = self.shortcut(inputs)
        x = self.relu(x1 + x2)
        return x


class DilatedConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(DilatedConv, self).__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3,
                      stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3,
                      stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3,
                      stride=1, padding=6, dilation=6),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3,
                      stride=1, padding=9, dilation=9),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

        self.c5 = nn.Sequential(
            nn.Conv2d(out_c*4, out_c, kernel_size=1,
                      stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.c5(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x0 * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, stride=1,
                               padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x0 * self.sigmoid(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(DecoderBlock, self).__init__()

        self.up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True)
        self.r1 = ResidualBlock(in_c[0] + in_c[1], out_c)
        self.r2 = ResidualBlock(out_c, out_c)

        self.ca = ChannelAttention(out_c)
        self.sa = SpatialAttention()

    def forward(self, x, s):
        x = self.up(x)
        x = torch.cat([x, s], dim=1)
        x = self.r1(x)
        x = self.r2(x)

        x = self.ca(x)
        x = self.sa(x)

        return x


class RUPNet(nn.Module):
    def __init__(self, feature_extractor_name):
        super(RUPNet, self).__init__()
        self.feature_extractor = timm.create_model(
            feature_extractor_name,
            pretrained=True,
            features_only=True
        )

        """ Dilated Conv + Pooling """
        self.r1 = nn.Sequential(DilatedConv(64, 64), nn.MaxPool2d((8, 8)))
        self.r2 = nn.Sequential(DilatedConv(256, 64), nn.MaxPool2d((4, 4)))
        self.r3 = nn.Sequential(DilatedConv(512, 64), nn.MaxPool2d((2, 2)))
        self.r4 = DilatedConv(1024, 64)

        """ Decoder """
        self.d1 = DecoderBlock([256, 512], 256)
        self.d2 = DecoderBlock([256, 256], 128)
        self.d3 = DecoderBlock([128, 64], 64)
        self.d4 = DecoderBlock([64, 3], 32)

        """ Output """
        self.y = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())

    def forward(self, x):
        s0 = x
        features = self.feature_extractor(x)
        s1, s2, s3, s4 = features[:-1]

        """ Dilated Conv + Pooling """
        r1 = self.r1(s1)
        r2 = self.r2(s2)
        r3 = self.r3(s3)
        r4 = self.r4(s4)

        rx = torch.cat([r1, r2, r3, r4], dim=1)

        """ Decoder """
        d1 = self.d1(rx, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)
        d4 = self.d4(d3, s0)

        y = self.y(d4)
        return y
