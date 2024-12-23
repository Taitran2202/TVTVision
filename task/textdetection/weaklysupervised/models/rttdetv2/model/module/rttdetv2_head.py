import torch
import torch.nn as nn
import torch.nn.functional as F
from .rttdetv2_basic import Conv


class SegformerHead(nn.Module):
    def __init__(self, cfg, feats_dim, out_dim, num_classes=2):
        super().__init__()
        # --------- Basic Parameters ----------
        self.feats_dim = feats_dim
        self.out_dim = out_dim
        self.num_classes = num_classes

        self.convs = nn.ModuleList()
        for i in range(len(feats_dim)):
            self.convs.append(
                Conv(
                    self.feats_dim[i],
                    self.out_dim,
                    k=1,
                    p=0,
                    s=1,
                    act_type=cfg['head_act'],
                    norm_type=cfg['head_norm'],
                    depthwise=cfg['head_depthwise']
                )
            )

        self.fusion_conv = Conv(
            self.out_dim * len(feats_dim),
            self.out_dim,
            k=1,
            p=0,
            s=1,
            act_type=cfg['head_act'],
            norm_type=cfg['head_norm'],
            depthwise=cfg['head_depthwise']
        )

        self.cls_seg = nn.Conv2d(out_dim, num_classes, kernel_size=1)

    def forward(self, inputs):
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                F.interpolate(
                    conv(x),
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=False)
            )
        out = self.fusion_conv(torch.cat(outs, dim=1))
        out = self.cls_seg(out)

        return out


def build_head(cfg, feats_dim, out_dim, num_classes=2):
    head = SegformerHead(cfg, feats_dim, out_dim, num_classes)

    return head
