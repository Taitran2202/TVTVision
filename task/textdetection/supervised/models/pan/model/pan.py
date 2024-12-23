import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from .module import resnet18, FPEM, Conv_BN_ReLU, FFM, PA_Head


class PAN(nn.Module):
    def __init__(self, pretrained, neck_channel, pa_in_channels, hidden_dim, num_classes):
        super(PAN, self).__init__()
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.backbone = resnet18(pretrained=pretrained)
        in_channels = neck_channel
        self.reduce_layer1 = Conv_BN_ReLU(in_channels[0], 128)
        self.reduce_layer2 = Conv_BN_ReLU(in_channels[1], 128)
        self.reduce_layer3 = Conv_BN_ReLU(in_channels[2], 128)
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[3], 128)

        self.fpem1 = FPEM(128, 128)
        self.fpem2 = FPEM(128, 128)

        self.ffm = FFM()

        self.det_head = PA_Head(pa_in_channels, hidden_dim, num_classes)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self, imgs):
        imgs = self.normalize(imgs)
        # backbone
        f = self.backbone(imgs)

        # reduce channel
        f1 = self.reduce_layer1(f[0])
        f2 = self.reduce_layer2(f[1])
        f3 = self.reduce_layer3(f[2])
        f4 = self.reduce_layer4(f[3])

        # FPEM
        f1_1, f2_1, f3_1, f4_1 = self.fpem1(f1, f2, f3, f4)
        f1_2, f2_2, f3_2, f4_2 = self.fpem2(f1_1, f2_1, f3_1, f4_1)

        # FFM
        f = self.ffm(f1_1, f2_1, f3_1, f4_1, f1_2, f2_2, f3_2, f4_2)

        # detection
        det_out = self.det_head(f)
        if self.training:
            det_out = self._upsample(det_out, imgs.size(), scale=1)
            return det_out
        else:
            det_out = self._upsample(det_out, imgs.size(), scale=4)
            score_out = torch.sigmoid(det_out[:, 0, :, :])

            return det_out, score_out
