import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms as transforms
from .module import build_backbone


class DoubleConvBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(DoubleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CRAFT(nn.Module):
    def __init__(self, backbone: str = 'resnet50', pretrained: bool = True):
        super(CRAFT, self).__init__()
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        self.backbone, self.backbone_dims = build_backbone(
            backbone=backbone, pretrained=pretrained, freeze_bn=False)

        self.upconv1 = DoubleConvBlock(
            self.backbone_dims[4], self.backbone_dims[3], self.backbone_dims[2])
        self.upconv2 = DoubleConvBlock(
            self.backbone_dims[2], self.backbone_dims[2], self.backbone_dims[1])
        self.upconv3 = DoubleConvBlock(
            self.backbone_dims[1], self.backbone_dims[1], self.backbone_dims[0])
        self.upconv4 = DoubleConvBlock(
            self.backbone_dims[0], self.backbone_dims[0], 32)

        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1),
        )

        for module in [self.upconv1, self.upconv2, self.upconv3, self.upconv4, self.conv_cls]:
            self.init_weights(module.modules())

    def init_weights(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, inputs):
        # normalize
        inputs = self.normalize(inputs)

        # extract features
        features = self.backbone(inputs)
        y = F.interpolate(features[4], size=features[3].size()[
                          2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, features[3]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=features[2].size()[
                          2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, features[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=features[1].size()[
                          2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, features[1]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=features[0].size()[
                          2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, features[0]], dim=1)

        feature = self.upconv4(y)

        outputs = self.conv_cls(feature)

        return outputs[:, 0, ...], outputs[:, 1, ...]
