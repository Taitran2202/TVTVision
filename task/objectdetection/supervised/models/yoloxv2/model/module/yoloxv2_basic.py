import torch
import torch.nn as nn


def get_conv2d(c1, c2, k, p, s, d, g, bias=False):
    conv = nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias)

    return conv

def get_activation(act_type=None):
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type == 'mish':
        return nn.Mish(inplace=True)
    elif act_type == 'silu':
        return nn.SiLU(inplace=True)
    elif act_type is None:
        return nn.Identity()
    else:
        raise NotImplementedError
        
def get_norm(norm_type, dim):
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=dim)
    elif norm_type is None:
        return nn.Identity()
    else:
        raise NotImplementedError

class Conv(nn.Module):
    def __init__(self, 
                 c1,                   # in channels
                 c2,                   # out channels 
                 k=1,                  # kernel size 
                 p=0,                  # padding
                 s=1,                  # padding
                 d=1,                  # dilation
                 act_type='lrelu',     # activation
                 norm_type='BN',       # normalization
                 depthwise=False):
        super(Conv, self).__init__()
        convs = []
        add_bias = False if norm_type else True
        if depthwise:
            convs.append(get_conv2d(c1, c1, k=k, p=p, s=s, d=d, g=c1, bias=add_bias))
            # depthwise conv
            if norm_type:
                convs.append(get_norm(norm_type, c1))
            if act_type:
                convs.append(get_activation(act_type))
            # pointwise conv
            convs.append(get_conv2d(c1, c2, k=1, p=0, s=1, d=d, g=1, bias=add_bias))
            if norm_type:
                convs.append(get_norm(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))

        else:
            convs.append(get_conv2d(c1, c2, k=k, p=p, s=s, d=d, g=1, bias=add_bias))
            if norm_type:
                convs.append(get_norm(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))
            
        self.convs = nn.Sequential(*convs)


    def forward(self, x):
        return self.convs(x)


# --------------------- Yolov8 modules ---------------------
## Yolo BottleNeck
class YoloBottleneck(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio = 0.5,
                 kernel_sizes = [3, 3],
                 shortcut     = True,
                 act_type     = 'silu',
                 norm_type    = 'BN',
                 depthwise    = False,):
        super(YoloBottleneck, self).__init__()
        inter_dim = int(out_dim * expand_ratio)  # hidden channels            
        self.cv1 = Conv(in_dim, inter_dim,  k=kernel_sizes[0], p=kernel_sizes[0]//2, norm_type=norm_type, act_type=act_type, depthwise=depthwise)
        self.cv2 = Conv(inter_dim, out_dim, k=kernel_sizes[1], p=kernel_sizes[1]//2, norm_type=norm_type, act_type=act_type, depthwise=depthwise)
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.cv2(self.cv1(x))

        return x + h if self.shortcut else h

## Yolo StageBlock
class Yolox2StageBlock(nn.Module):
    def __init__(self,
                 in_dim     :int,
                 out_dim    :int,
                 num_blocks :int  = 1,
                 shortcut   :bool = False,
                 act_type   :str  = 'silu',
                 norm_type  :str  = 'BN',
                 depthwise  :bool = False,):
        super(Yolox2StageBlock, self).__init__()
        self.inter_dim = out_dim // 2
        self.cv1 = Conv(in_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(in_dim, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.blocks = nn.Sequential(*(
            YoloBottleneck(self.inter_dim, self.inter_dim, 1.0, [1, 3], shortcut, act_type, norm_type, depthwise)
            for _ in range(num_blocks)))
        self.cv3 = Conv(self.inter_dim * num_blocks, self.inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.output_proj = Conv(2 * self.inter_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        # Input proj
        x1 = self.cv1(x)
        x2 = self.cv2(x)

        # Bottleneck
        out = []
        for m in self.blocks:
            x2 = m(x2)
            out.append(x2)
        x2 = self.cv3(torch.cat(out, dim=1))

        # Output proj
        out = self.output_proj(torch.cat([x1, x2], dim=1))

        return out
    