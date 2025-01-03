import torch
import torch.nn as nn
from .yoloxv2_basic import Conv, Yolox2StageBlock


# ---------------------------- Backbone ----------------------------
class Yolox2Backbone(nn.Module):
    def __init__(self, width=1.0, depth=1.0, act_type='silu', norm_type='BN', depthwise=False):
        super(Yolox2Backbone, self).__init__()
        self.feat_dims = [round(64 * width), round(128 * width),
                          round(256 * width), round(512 * width), round(1024 * width)]
        # P1/2
        self.layer_1 = Conv(
            3, self.feat_dims[0], k=6, p=2, s=2, act_type=act_type, norm_type=norm_type)
        # P2/4
        self.layer_2 = nn.Sequential(
            Conv(self.feat_dims[0], self.feat_dims[1], k=3,
                 p=1, s=2, act_type=act_type, norm_type=norm_type),
            Yolox2StageBlock(in_dim=self.feat_dims[1],
                             out_dim=self.feat_dims[1],
                             num_blocks=round(3*depth),
                             shortcut=True,
                             act_type=act_type,
                             norm_type=norm_type,
                             depthwise=depthwise)
        )
        # P3/8
        self.layer_3 = nn.Sequential(
            Conv(self.feat_dims[1], self.feat_dims[2], k=3,
                 p=1, s=2, act_type=act_type, norm_type=norm_type),
            Yolox2StageBlock(in_dim=self.feat_dims[2],
                             out_dim=self.feat_dims[2],
                             num_blocks=round(9*depth),
                             shortcut=True,
                             act_type=act_type,
                             norm_type=norm_type,
                             depthwise=depthwise)
        )
        # P4/16
        self.layer_4 = nn.Sequential(
            Conv(self.feat_dims[2], self.feat_dims[3], k=3,
                 p=1, s=2, act_type=act_type, norm_type=norm_type),
            Yolox2StageBlock(in_dim=self.feat_dims[3],
                             out_dim=self.feat_dims[3],
                             num_blocks=round(9*depth),
                             shortcut=True,
                             act_type=act_type,
                             norm_type=norm_type,
                             depthwise=depthwise)
        )
        # P5/32
        self.layer_5 = nn.Sequential(
            Conv(self.feat_dims[3], self.feat_dims[4], k=3,
                 p=1, s=2, act_type=act_type, norm_type=norm_type),
            Yolox2StageBlock(in_dim=self.feat_dims[4],
                             out_dim=self.feat_dims[4],
                             num_blocks=round(3*depth),
                             shortcut=True,
                             act_type=act_type,
                             norm_type=norm_type,
                             depthwise=depthwise)
        )

    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        outputs = [c3, c4, c5]

        return outputs


# ---------------------------- Functions ----------------------------
# build Backbone
def build_backbone(cfg):
    # model
    backbone = Yolox2Backbone(width=cfg.width,
                              depth=cfg.depth,
                              act_type=cfg.bk_act,
                              norm_type=cfg.bk_norm,
                              depthwise=cfg.bk_depthwise
                              )
    feat_dims = backbone.feat_dims[-3:]

    return backbone, feat_dims
