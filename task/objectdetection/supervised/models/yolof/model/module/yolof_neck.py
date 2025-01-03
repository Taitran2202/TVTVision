import torch.nn as nn

from .yolof_basic import Conv


def c2_xavier_fill(module: nn.Module):
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


class Bottleneck(nn.Module):
    def __init__(self, in_dim, dilation, expand_ratio, act_type='relu', norm_type='BN'):
        super(Bottleneck, self).__init__()
        # ------------------ Basic parameters -------------------
        self.in_dim = in_dim
        self.dilation = dilation
        self.expand_ratio = expand_ratio
        inter_dim = round(in_dim * expand_ratio)
        # ------------------ Network parameters -------------------
        self.branch = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(inter_dim, inter_dim, k=3, p=dilation, d=dilation,
                 act_type=act_type, norm_type=norm_type),
            Conv(inter_dim, in_dim, k=1, act_type=act_type, norm_type=norm_type)
        )

    def forward(self, x):
        return x + self.branch(x)


class DilatedEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, expand_ratio, dilations=[2, 4, 6, 8], act_type='relu', norm_type='BN'):
        super(DilatedEncoder, self).__init__()
        # ------------------ Basic parameters -------------------
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.expand_ratio = expand_ratio
        self.dilations = dilations
        # ------------------ Network parameters -------------------
        # proj layer
        self.projector = nn.Sequential(
            Conv(in_dim, out_dim, k=1, act_type=None, norm_type=norm_type),
            Conv(out_dim, out_dim, k=3, p=1, act_type=None, norm_type=norm_type)
        )
        # encoder layers
        self.encoders = nn.Sequential(
            *[Bottleneck(out_dim, d, expand_ratio, act_type, norm_type) for d in dilations])

        self._init_weight()

    def _init_weight(self):
        for m in self.projector:
            if isinstance(m, nn.Conv2d):
                c2_xavier_fill(m)
                c2_xavier_fill(m)
            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.encoders.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.projector(x)
        x = self.encoders(x)

        return x


def build_neck(cfg, in_dim, out_dim):
    model = DilatedEncoder(
        in_dim=in_dim,
        out_dim=out_dim,
        expand_ratio=cfg.neck_expand_ratio,
        dilations=cfg.neck_dilations,
        act_type=cfg.neck_act,
        norm_type=cfg.neck_norm
    )
    return model
