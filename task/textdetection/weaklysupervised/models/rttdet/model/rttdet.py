import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from .module import *


class RTTDet(nn.Module):
    def __init__(self, cfg, num_classes=2):
        super(RTTDet, self).__init__()
        # ---------------------- Basic Parameters ----------------------
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.cfg = cfg
        self.num_classes = num_classes

        # ------------------- Network Structure -------------------
        # backbone network
        self.backbone, feats_dim = build_backbone(
            cfg, cfg['pretrained'])

        # detection head
        self.decode_head = build_head(
            cfg, feats_dim, feats_dim[0], self.num_classes)

    def forward(self, x):
        x = self.normalize(x)

        # backbone
        feats = self.backbone(x)

        # predict
        outputs = self.decode_head(feats)
        outputs = F.interpolate(outputs, scale_factor=2,
                                mode='bilinear', align_corners=False)

        return outputs[:, 0, ...], outputs[:, 1, ...]
