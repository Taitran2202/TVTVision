import torch
import torch.nn as nn
import torchvision.transforms as transforms

model_urls = {
    "elancspnet_nano": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elan_cspnet_nano.pth",
    "elancspnet_small": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elan_cspnet_small.pth",
    "elancspnet_medium": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elan_cspnet_medium.pth",
    "elancspnet_large": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/elan_cspnet_large.pth",
    "elancspnet_huge": None,
}


def get_activation(act_type=None):
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type == 'mish':
        return nn.Mish(inplace=True)
    elif act_type == 'silu':
        return nn.SiLU(inplace=True)


def get_norm(norm_type, dim):
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=dim)


# Basic conv layer
class Conv(nn.Module):
    def __init__(self,
                 c1,                   # in channels
                 c2,                   # out channels
                 k=1,                  # kernel size
                 p=0,                  # padding
                 s=1,                  # padding
                 d=1,                  # dilation
                 act_type='silu',      # activation
                 norm_type='BN',       # normalization
                 depthwise=False):
        super(Conv, self).__init__()
        convs = []
        if depthwise:
            # depthwise conv
            convs.append(nn.Conv2d(c1, c1, kernel_size=k, stride=s,
                         padding=p, dilation=d, groups=c1, bias=False))
            convs.append(get_norm(norm_type, c1))
            if act_type is not None:
                convs.append(get_activation(act_type))

            # pointwise conv
            convs.append(nn.Conv2d(c1, c2, kernel_size=1, stride=1,
                         padding=0, dilation=d, groups=1, bias=False))
            convs.append(get_norm(norm_type, c2))
            if act_type is not None:
                convs.append(get_activation(act_type))

        else:
            convs.append(nn.Conv2d(c1, c2, kernel_size=k, stride=s,
                         padding=p, dilation=d, groups=1, bias=False))
            convs.append(get_norm(norm_type, c2))
            if act_type is not None:
                convs.append(get_activation(act_type))

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)


# BottleNeck
class Bottleneck(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio=0.5,
                 shortcut=False,
                 depthwise=False,
                 act_type='silu',
                 norm_type='BN'):
        super(Bottleneck, self).__init__()
        inter_dim = int(out_dim * expand_ratio)  # hidden channels
        self.cv1 = Conv(in_dim, inter_dim, k=3, p=1, norm_type=norm_type,
                        act_type=act_type, depthwise=depthwise)
        self.cv2 = Conv(inter_dim, out_dim, k=3, p=1,
                        norm_type=norm_type, act_type=act_type, depthwise=depthwise)
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.cv2(self.cv1(x))

        return x + h if self.shortcut else h


# ELAN-CSP-Block
class ELAN_CSP_Block(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio=0.5,
                 nblocks=1,
                 shortcut=False,
                 depthwise=False,
                 act_type='silu',
                 norm_type='BN'):
        super(ELAN_CSP_Block, self).__init__()
        inter_dim = int(out_dim * expand_ratio)
        self.cv1 = Conv(in_dim, inter_dim, k=1,
                        norm_type=norm_type, act_type=act_type)
        self.cv2 = Conv(in_dim, inter_dim, k=1,
                        norm_type=norm_type, act_type=act_type)
        self.m = nn.Sequential(*(
            Bottleneck(inter_dim, inter_dim, 1.0, shortcut,
                       depthwise, act_type, norm_type)
            for _ in range(nblocks)))
        self.cv3 = Conv((2 + nblocks) * inter_dim, out_dim, k=1,
                        act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        out = list([x1, x2])

        out.extend(m(out[-1]) for m in self.m)

        out = self.cv3(torch.cat(out, dim=1))

        return out


# ELAN-CSPNet
class ELAN_CSPNet(nn.Module):
    """ YOLOv8' backbone """

    def __init__(self, width=1.0, depth=1.0, ratio=1.0, act_type='silu', norm_type='BN', depthwise=False, num_classes=1000):
        super(ELAN_CSPNet, self).__init__()
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        # stride = 2
        self.layer_1 = Conv(3, int(64*width), k=3, p=1, s=2,
                            act_type=act_type, norm_type=norm_type)

        # stride = 4
        self.layer_2 = nn.Sequential(
            Conv(int(64*width), int(128*width), k=3, p=1,
                 s=2, act_type=act_type, norm_type=norm_type),
            ELAN_CSP_Block(int(128*width), int(128*width), nblocks=int(3*depth), shortcut=True,
                           act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # stride = 8
        self.layer_3 = nn.Sequential(
            Conv(int(128*width), int(256*width), k=3, p=1,
                 s=2, act_type=act_type, norm_type=norm_type),
            ELAN_CSP_Block(int(256*width), int(256*width), nblocks=int(6*depth), shortcut=True,
                           act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # stride = 16
        self.layer_4 = nn.Sequential(
            Conv(int(256*width), int(512*width), k=3, p=1,
                 s=2, act_type=act_type, norm_type=norm_type),
            ELAN_CSP_Block(int(512*width), int(512*width), nblocks=int(6*depth), shortcut=True,
                           act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )
        # stride = 32
        self.layer_5 = nn.Sequential(
            Conv(int(512*width), int(512*width*ratio), k=3, p=1,
                 s=2, act_type=act_type, norm_type=norm_type),
            ELAN_CSP_Block(int(512*width*ratio), int(512*width*ratio), nblocks=int(3*depth), shortcut=True,
                           act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512*width*ratio), num_classes)

    def forward(self, x):
        x = self.normalize(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

        # [B, C, H, W] -> [B, C, 1, 1]
        x = self.avgpool(x)
        # [B, C, 1, 1] -> [B, C]
        x = x.flatten(1)
        x = self.fc(x)

        return x


# build ELAN-CSPNet
def build_elan_cspnet(model_name='large', pretrained=False, num_classes=None):
    # model
    if model_name == 'huge':
        model = ELAN_CSPNet(width=1.25, depth=1.0, ratio=1.0,
                            act_type='silu', norm_type='BN')
    elif model_name == 'large':
        model = ELAN_CSPNet(width=1.0, depth=1.0, ratio=1.0,
                            act_type='silu', norm_type='BN')
    elif model_name == 'medium':
        model = ELAN_CSPNet(width=0.75, depth=0.67, ratio=1.5,
                            act_type='silu', norm_type='BN')
    elif model_name == 'small':
        model = ELAN_CSPNet(width=0.5, depth=0.34, ratio=2.0,
                            act_type='silu', norm_type='BN')
    elif model_name == 'nano':
        model = ELAN_CSPNet(width=0.25, depth=0.34, ratio=2.0,
                            act_type='silu', norm_type='BN')
        
    url = model_urls['elancspnet_' + model_name]

    # load weight
    if pretrained and url is not None:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                checkpoint_state_dict.pop(k)

        model.load_state_dict(checkpoint_state_dict)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model