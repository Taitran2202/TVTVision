import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from .module import resnet50, FPN, PSENet_Head


class PSENet(nn.Module):
    def __init__(self, pretrained, neck_in_channel, neck_out_channel, pse_in_channels, hidden_dim, num_classes):
        super(PSENet, self).__init__()
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.backbone = resnet50(pretrained=pretrained)
        self.fpn = FPN(in_channels=neck_in_channel,
                       out_channels=neck_out_channel)
        self.det_head = PSENet_Head(
            in_channels=pse_in_channels, hidden_dim=hidden_dim, num_classes=num_classes)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self, imgs):
        imgs = self.normalize(imgs)
        # backbone
        f = self.backbone(imgs)

        # FPN
        f1, f2, f3, f4, = self.fpn(f[0], f[1], f[2], f[3])

        f = torch.cat((f1, f2, f3, f4), 1)

        # detection
        det_out = self.det_head(f)
        det_out = self._upsample(det_out, imgs.size(), scale=1)
        if not self.training:
            score_out = torch.sigmoid(det_out[:, 0, :, :])
            return det_out, score_out
        
        return det_out
