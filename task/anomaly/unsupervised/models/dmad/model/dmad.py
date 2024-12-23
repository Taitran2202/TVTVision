import torch.nn as nn
import torchvision.transforms as transforms
from .module import *


class DMAD(nn.Module):
    def __init__(self, backbone, input_size: tuple[int, int], vq: bool, gamma: float, amap_mode: str):
        super(DMAD, self).__init__()
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.input_size = input_size
        self.backbone = backbone

        self.encoder, self.bn, self.offset = eval(
            f'{backbone}(pretrained=True, vq={vq}, gamma={gamma})')
        self.decoder = eval(f'de_{backbone}(pretrained=False)')

        self.anomaly_map_generator = AnomalyMapGenerator(
            input_size=input_size, amap_mode=amap_mode)
        
        for parameters in self.encoder.parameters():
            parameters.requires_grad = False
        self.encoder.eval()

    def forward(self, x):
        x = self.normalize(x)
        if self.training:
            self.offset.train()
            _, img_, offset_loss = self.offset(x)
            inputs = self.encoder(img_)
            vq, vq_loss = self.bn(inputs)
            outputs = self.decoder(vq)
            
            return inputs, outputs, vq_loss, offset_loss
        else:
            self.offset.eval()
            img_, grid1_, grid2_ = self.offset(x)
            inputs = self.encoder(img_)
            vq, _ = self.bn(inputs)
            outputs = self.decoder(vq)
            
            return self.anomaly_map_generator(inputs, outputs, grid1_, grid2_)[:, 0, ...]
