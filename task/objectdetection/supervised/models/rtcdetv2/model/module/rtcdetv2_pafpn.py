import torch
import torch.nn as nn
import torch.nn.functional as F
from .rtcdetv2_basic import (Conv, build_reduce_layer,
                             build_downsample_layer, build_fpn_block)


# RTCDet-Style PaFPN
class RTCDetPaFPN(nn.Module):
    def __init__(self, cfg, in_dims=[512, 1024, 512], out_dim=None):
        super(RTCDetPaFPN, self).__init__()
        # --------------------------- Basic Parameters ---------------------------
        self.in_dims = in_dims

        # --------------------------- Top-down FPN ---------------------------
        # P5 -> P4
        self.reduce_layer_1 = build_reduce_layer(
            cfg, in_dims[2], round(512*cfg.width))
        self.reduce_layer_2 = build_reduce_layer(
            cfg, in_dims[1], round(512*cfg.width))
        self.top_down_layer_1 = build_fpn_block(cfg, round(
            512*cfg.width) + round(512*cfg.width), round(512*cfg.width))

        # P4 -> P3
        self.reduce_layer_3 = build_reduce_layer(
            cfg, round(512*cfg.width), round(256*cfg.width))
        self.reduce_layer_4 = build_reduce_layer(
            cfg, in_dims[0], round(256*cfg.width))
        self.top_down_layer_2 = build_fpn_block(cfg, round(
            256*cfg.width) + round(256*cfg.width), round(256*cfg.width))

        # --------------------------- Bottom-up FPN ---------------------------
        # P3 -> P4
        self.downsample_layer_1 = build_downsample_layer(
            cfg, round(256*cfg.width), round(256*cfg.width))
        self.bottom_up_layer_1 = build_fpn_block(cfg, round(
            256*cfg.width) + round(256*cfg.width), round(512*cfg.width))

        # P4 -> P5
        self.downsample_layer_2 = build_downsample_layer(
            cfg, round(512*cfg.width), round(512*cfg.width))
        self.bottom_up_layer_2 = build_fpn_block(cfg, round(
            512*cfg.width) + round(512*cfg.width), round(1024*cfg.width))

        # --------------------------- Output proj ---------------------------
        if out_dim is not None:
            self.out_layers = nn.ModuleList([
                Conv(in_dim, out_dim, k=1,
                     act_type=cfg.fpn_act, norm_type=cfg.fpn_norm)
                for in_dim in [round(256*cfg.width), round(512*cfg.width), round(1024*cfg.width)]])
            self.out_dim = [out_dim] * 3
        else:
            self.out_layers = None
            self.out_dim = [
                round(256*cfg.width), round(512*cfg.width), round(1024*cfg.width)]

    def forward(self, fpn_feats):
        c3, c4, c5 = fpn_feats

        # Top down
        # P5 -> P4
        c6 = self.reduce_layer_1(c5)
        c7 = self.reduce_layer_2(c4)
        c8 = torch.cat([F.interpolate(c6, scale_factor=2.0), c7], dim=1)
        c9 = self.top_down_layer_1(c8)
        # P4 -> P3
        c10 = self.reduce_layer_3(c9)
        c11 = self.reduce_layer_4(c3)
        c12 = torch.cat([F.interpolate(c10, scale_factor=2.0), c11], dim=1)
        c13 = self.top_down_layer_2(c12)

        # Bottom up
        # p3 -> P4
        c14 = self.downsample_layer_1(c13)
        c15 = torch.cat([c14, c10], dim=1)
        c16 = self.bottom_up_layer_1(c15)
        # P4 -> P5
        c17 = self.downsample_layer_2(c16)
        c18 = torch.cat([c17, c6], dim=1)
        c19 = self.bottom_up_layer_2(c18)

        out_feats = [c13, c16, c19]  # [P3, P4, P5]

        # output proj layers
        if self.out_layers is not None:
            out_feats_proj = []
            for feat, layer in zip(out_feats, self.out_layers):
                out_feats_proj.append(layer(feat))
            return out_feats_proj

        return out_feats


def build_fpn(cfg, in_dims, out_dim=None):
    model = cfg.fpn
    # build pafpn
    if model == 'rtcdet_pafpn':
        fpn_net = RTCDetPaFPN(cfg, in_dims, out_dim)

    return fpn_net
