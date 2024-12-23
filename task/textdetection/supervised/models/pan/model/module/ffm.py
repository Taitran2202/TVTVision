import torch
import torch.nn as nn
import torch.nn.functional as F


class FFM(nn.Module):
    def __init__(self):
        super(FFM, self).__init__()

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self, f1_1, f2_1, f3_1, f4_1, f1_2, f2_2, f3_2, f4_2):
        f1 = f1_1 + f1_2
        f2 = f2_1 + f2_2
        f3 = f3_1 + f3_2
        f4 = f4_1 + f4_2
        f2 = self._upsample(f2, f1.size())
        f3 = self._upsample(f3, f1.size())
        f4 = self._upsample(f4, f1.size())
        f = torch.cat((f1, f2, f3, f4), 1)

        return f
