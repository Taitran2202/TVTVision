import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torchvision.models import resnet

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

ARCH = {
    "resnet18": (resnet.BasicBlock, (2, 2, 2, 2), (64, 64, 128, 256, 512)),
    "resnet34": (resnet.BasicBlock, (3, 4, 6, 3), (64, 64, 128, 256, 512)),
    "resnet50": (resnet.Bottleneck, (3, 4, 6, 3), (64, 256, 512, 1024, 2048)),
    "resnet101": (resnet.Bottleneck, (3, 4, 23, 3), (64, 256, 512, 1024, 2048)),
}


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()

        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.eps = eps
        self.num_features = num_features

    def forward(self, input):
        if input.requires_grad:
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            dtype = input.dtype

            return input * scale.to(dtype) + bias.to(dtype)

        else:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module

        if isinstance(module, bn_module):
            res = cls(module.num_features)

            if module.affine:
                res.weight.detach().copy_(module.weight.detach())
                res.bias.detach().copy_(module.bias.detach())

            res.running_mean.detach().copy_(module.running_mean.detach())
            res.running_var.detach().copy_(module.running_var.detach())
            res.eps = module.eps

        else:
            for n, c in module.named_children():
                new_c = cls.convert_frozen_batchnorm(c)

                if new_c is not c:
                    res.add_module(n, new_c)

        return res

    def __repr__(self):
        return f"FrozenBatchNorm2d(num_features={self.num_features}, eps={self.eps})"


class ResNetFeature(resnet.ResNet):
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        stride2 = self.relu(x)
        stride4 = self.maxpool(stride2)

        stride4 = self.layer1(stride4)
        stride8 = self.layer2(stride4)
        stride16 = self.layer3(stride8)
        stride32 = self.layer4(stride16)

        return (stride2, stride4, stride8, stride16, stride32)


def build_backbone(backbone, pretrained=True, freeze_bn=True):

    block, layers, dim = ARCH.get(backbone)

    model = ResNetFeature(block=block, layers=layers)

    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[backbone], progress=True)

        model.load_state_dict(state_dict)

        if freeze_bn:
            model = FrozenBatchNorm2d.convert_frozen_batchnorm(model)

    return model, dim
