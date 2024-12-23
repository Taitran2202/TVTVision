import timm
import torch
import torch.nn as nn


class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=False, act=True):
        super(Conv2D, self).__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c, out_c,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResidualBlock, self).__init__()

        self.network = nn.Sequential(
            Conv2D(in_c, out_c),
            Conv2D(out_c, out_c, kernel_size=1, padding=0, act=False)

        )
        self.shortcut = Conv2D(
            in_c, out_c, kernel_size=1, padding=0, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_init):
        x = self.network(x_init)
        s = self.shortcut(x_init)
        x = self.relu(x + s)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class EncoderBlock(nn.Module):
    def __init__(self, ch, feature_extractor_name):
        super(EncoderBlock, self).__init__()

        """ ResNet50 """
        self.feature_extractor = timm.create_model(
            feature_extractor_name,
            pretrained=True,
            features_only=True
        )

        """ Reduce feature channels """
        self.c1 = Conv2D(64, ch)
        self.c2 = Conv2D(256, ch)
        self.c3 = Conv2D(512, ch)
        self.c4 = Conv2D(1024, ch)

    def forward(self, x):
        """ Backbone: ResNet50 """
        features = self.feature_extractor(x)
        x1, x2, x3, x4 = features[:-1]

        c1 = self.c1(x1)
        c2 = self.c2(x2)
        c3 = self.c3(x3)
        c4 = self.c4(x4)

        return c1, c2, c3, c4


class MultiKernel_Dilated_Conv(nn.Module):
    def __init__(self, in_c, out_c):
        super(MultiKernel_Dilated_Conv, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.c1 = Conv2D(in_c, out_c, kernel_size=1, padding=0)
        self.c2 = Conv2D(in_c, out_c, kernel_size=3, padding=1)
        self.c3 = Conv2D(in_c, out_c, kernel_size=7, padding=3)
        self.c4 = Conv2D(in_c, out_c, kernel_size=11, padding=5)
        self.s1 = Conv2D(out_c*4, out_c, kernel_size=1, padding=0)

        self.d1 = Conv2D(out_c, out_c, kernel_size=3, padding=1, dilation=1)
        self.d2 = Conv2D(out_c, out_c, kernel_size=3, padding=3, dilation=3)
        self.d3 = Conv2D(out_c, out_c, kernel_size=3, padding=7, dilation=7)
        self.d4 = Conv2D(out_c, out_c, kernel_size=3, padding=11, dilation=11)
        self.s2 = Conv2D(out_c*4, out_c, kernel_size=1, padding=0, act=False)
        self.s3 = Conv2D(in_c, out_c, kernel_size=1, padding=0, act=False)

        self.ca = ChannelAttention(out_c)
        self.sa = SpatialAttention()

    def forward(self, x):
        x0 = x
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        x = torch.cat([x1, x2, x3, x4], axis=1)
        x = self.s1(x)

        x1 = self.d1(x)
        x2 = self.d2(x)
        x3 = self.d3(x)
        x4 = self.d4(x)
        x = torch.cat([x1, x2, x3, x4], axis=1)
        x = self.s2(x)
        s = self.c3(x0)

        x = self.relu(x+s)
        x = x * self.ca(x)
        x = x * self.sa(x)

        return x


class MultiScale_Feature_Fusion(nn.Module):
    def __init__(self, in_c, out_c):
        super(MultiScale_Feature_Fusion, self).__init__()

        self.up_2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True)

        self.c1 = Conv2D(in_c, out_c)
        self.c2 = Conv2D(out_c+in_c, out_c)
        self.c3 = Conv2D(in_c, out_c)
        self.c4 = Conv2D(out_c+in_c, out_c)

        self.ca = ChannelAttention(out_c)
        self.sa = SpatialAttention()

    def forward(self, f1, f2, f3):
        x1 = self.up_2(f1)
        x1 = self.c1(x1)
        x1 = torch.cat([x1, f2], axis=1)
        x1 = self.up_2(x1)
        x1 = self.c2(x1)
        x1 = torch.cat([x1, f3], axis=1)
        x1 = self.up_2(x1)
        x1 = self.c4(x1)

        x1 = x1 * self.ca(x1)
        x1 = x1 * self.sa(x1)

        return x1


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True)
        self.r1 = ResidualBlock(in_c[0]+in_c[1], out_c)
        self.r2 = ResidualBlock(out_c, out_c)

    def forward(self, x, s):
        x = self.up(x)
        x = torch.cat([x, s], axis=1)
        x = self.r1(x)
        x = self.r2(x)
        return x


class DeepSegNet(nn.Module):
    def __init__(self, feature_extractor_name='resnet50'):
        super(DeepSegNet, self).__init__()

        """ Encoder """
        self.encoder = EncoderBlock(96, feature_extractor_name)

        """ MultiKernel Conv + Dilation """
        self.c1 = MultiKernel_Dilated_Conv(96, 96)
        self.c2 = MultiKernel_Dilated_Conv(96, 96)
        self.c3 = MultiKernel_Dilated_Conv(96, 96)
        self.c4 = MultiKernel_Dilated_Conv(96, 96)

        """ Decoder """
        self.d1 = DecoderBlock([96, 96], 96)
        self.d2 = DecoderBlock([96, 96], 96)
        self.d3 = DecoderBlock([96, 96], 96)

        """ Multiscale Feature Fusion """
        self.msf = MultiScale_Feature_Fusion(96, 96)

        """ Output """
        self.y = nn.Sequential(
            nn.Conv2d(96, 1, kernel_size=1, padding=0), nn.Sigmoid())

    def forward(self, image):
        s1, s2, s3, s4 = self.encoder(image)

        x1 = self.c1(s1)
        x2 = self.c2(s2)
        x3 = self.c3(s3)
        x4 = self.c4(s4)

        d1 = self.d1(x4, x3)
        d2 = self.d2(d1, x2)
        d3 = self.d3(d2, x1)

        x = self.msf(d1, d2, d3)
        y = self.y(x)

        return y
