import torch
import torch.nn as nn
import torch.nn.functional as F
from .coordconv import CoordConv2d


class Descriptor(nn.Module):
    """Descriptor module."""

    def __init__(self, gamma_d: int, backbone: str) -> None:
        super(Descriptor, self).__init__()

        self.backbone = backbone
        if self.backbone not in ["vgg19_bn", "resnet18", "wide_resnet50_2", "efficientnet_b5"]:
            raise ValueError(
                f"Supported backbones are [vgg19_bn, resnet18, wide_resnet50_2, efficientnet_b5]. Got {self.backbone} instead.")

        # TODO: Automatically infer the number of dims
        backbone_dims = {"vgg19_bn": 1280, "resnet18": 448,
                         "wide_resnet50_2": 1792, "efficientnet_b5": 568}

        dim = backbone_dims[backbone]
        out_channels = 2 * dim // gamma_d if backbone == "efficientnet_b5" else dim // gamma_d

        self.layer = CoordConv2d(
            in_channels=dim, out_channels=out_channels, kernel_size=1)

    def forward(self, features):
        patch_features = None
        for i in features:
            i = F.avg_pool2d(
                i, 3, 1, 1) / i.size(1) if self.backbone == "efficientnet_b5" else F.avg_pool2d(i, 3, 1, 1)
            patch_features = (
                i
                if patch_features is None
                else torch.cat((patch_features, F.interpolate(i, patch_features.size(2), mode="bilinear")), dim=1)
            )

        target_oriented_features = self.layer(patch_features)
        return target_oriented_features
