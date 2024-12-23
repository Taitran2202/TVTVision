import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

BUILDER = {
    'resnet18': models.resnet18,
    'resnet50': models.resnet50
}


class ResNet(nn.Module):
    def __init__(self, backbone_type, out_stride, out_layers, pretrained=True):
        super(ResNet, self).__init__()
        self.out_stride = out_stride
        self.out_layers = out_layers
        base_dim = 64 if backbone_type == "resnet18" else 256
        self.resnet = BUILDER[backbone_type](pretrained=pretrained)

        children = list(self.resnet.children())
        # layer0: conv + bn + relu + pool
        self.layer0 = nn.Sequential(*children[:4])
        self.layer1 = children[4]
        self.layer2 = children[5]
        self.layer3 = children[6]
        self.layer4 = children[7]

        planes = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]
        self.out_dim = sum([planes[i - 1] for i in self.out_layers])

    def forward(self, x):
        x = self.layer0(x)  # out_stride: 4
        feat1 = self.layer1(x)  # out_stride: 4
        feat2 = self.layer2(feat1)  # out_stride: 8
        feat3 = self.layer3(feat2)  # out_stride: 16
        feat4 = self.layer4(feat3)  # out_stride: 32
        feats = [feat1, feat2, feat3, feat4]
        out_strides = [4, 8, 16, 32]
        feat_list = []
        for i in self.out_layers:
            scale_factor = out_strides[i - 1] / self.out_stride
            feat = feats[i - 1]
            feat = F.interpolate(
                feat, scale_factor=scale_factor, mode="bilinear")
            feat_list.append(feat)
        feat = torch.cat(feat_list, dim=1)
        return feat
