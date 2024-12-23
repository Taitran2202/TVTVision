import copy
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from .module import *


class ReContrast(nn.Module):
    def __init__(self, input_size: tuple[int, int], amap_mode: str):
        super(ReContrast, self).__init__()
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.encoder, self.bottleneck = wide_resnet50_2(pretrained=True)
        self.decoder = de_wide_resnet50_2(pretrained=False, output_conv=2)
        self.encoder_freeze = copy.deepcopy(self.encoder)
        for parameters in self.encoder_freeze.parameters():
            parameters.requires_grad = False

        self.encoder.layer4 = None
        self.encoder.fc = None
        self.encoder_freeze.layer4 = None
        self.encoder_freeze.fc = None

        self.anomaly_map_generator = AnomalyMapGenerator(
            input_size=input_size, amap_mode=amap_mode)

    def forward(self, x):
        x = self.normalize(x)
        en = self.encoder(x)
        with torch.no_grad():
            en_freeze = self.encoder_freeze(x)
        en_2 = [torch.cat([a, b], dim=0) for a, b in zip(en, en_freeze)]
        de = self.decoder(self.bottleneck(en_2))
        de = [a.chunk(dim=0, chunks=2) for a in de]
        de = [de[0][0], de[1][0], de[2][0], de[3][1], de[4][1], de[5][1]]

        if self.training:
            return en_freeze + en, de
        else:
            return self.anomaly_map_generator((en_freeze + en), de)[:, 0, ...]
