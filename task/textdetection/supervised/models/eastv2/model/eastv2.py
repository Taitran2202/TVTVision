import torch.nn as nn
import torchvision.transforms as transforms
from .module.backbone import *
from .module.neck import FPN
from .module.head import EASTHead


class EASTv2(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, inner_channels=256, scope=512):
        super(EASTv2, self).__init__()
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        if backbone in ('mobilenetv3_large', 'mobilenetv3_small'):
            self.extractor = MobileNetV3(backbone)
        elif backbone in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'deformable_resnet18', 'deformable_resnet50', 'resnet152'):
            self.extractor = eval(backbone)(pretrained=pretrained)
        else:
            raise ValueError(
                f"backbone {backbone} is not supported. List of available backbones are "
                "['MobileNetV3', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'deformable_resnet18', 'deformable_resnet50', 'resnet152']."
            )
        self.merge = FPN(in_channels=self.extractor.out_channels,
                         inner_channels=inner_channels)
        self.output = EASTHead(in_channels=inner_channels, scope=scope)

    def forward(self, x):
        x = self.normalize(x)
        return self.output(self.merge(self.extractor(x)))
