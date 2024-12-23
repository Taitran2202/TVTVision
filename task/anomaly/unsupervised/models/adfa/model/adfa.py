from tqdm import tqdm
from einops import rearrange
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import create_feature_extractor
from .module import Descriptor, AnomalyMapGenerator, EfficientNet as effnet


def get_return_nodes(backbone: str) -> list[str]:
    return_nodes: list[str]
    if backbone in ("resnet18", "wide_resnet50_2"):
        return_nodes = ["layer1", "layer2", "layer3"]
    elif backbone == "vgg19_bn":
        return_nodes = ["features.25", "features.38", "features.52"]
    elif backbone == "efficientnet_b5":
        return_nodes = ["reduction_2", "reduction_3", "reduction_4"]
    else:
        raise ValueError(
            f"Backbone {backbone} is not supported. Supported backbones are [vgg19_bn, resnet18, wide_resnet50_2, efficientnet_b5].")

    return return_nodes


def get_feature_extractor(backbone: str, return_nodes: list[str]):
    if backbone == 'efficientnet_b5':
        feature_extractor = effnet.from_pretrained('efficientnet-b5')
    else:
        model = getattr(models, backbone)(pretrained=True)
        feature_extractor = create_feature_extractor(
            model=model, return_nodes=return_nodes)
    feature_extractor.eval()

    return feature_extractor


def dryrun_find_featuremap_dims(feature_extractor, input_size: tuple[int, int], layers: list[str]):
    dryrun_input = torch.empty(1, 3, *input_size)
    dryrun_features = feature_extractor(dryrun_input)

    return {
        layer: {"num_features": dryrun_features[layer].shape[1],
                "resolution": dryrun_features[layer].shape[2:]}
        for layer in layers
    }


class ADFA(nn.Module):
    def __init__(self, input_size: tuple[int, int], backbone: str,
                 gamma_d: int, num_nearest_neighbors: int, radius: float):
        super(ADFA, self).__init__()
        self.input_size = torch.Size(input_size)
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.gamma_d = gamma_d

        self.num_nearest_neighbors = num_nearest_neighbors

        self.register_buffer("memory_bank", torch.tensor(0.0))
        self.memory_bank: Tensor

        return_nodes = get_return_nodes(backbone)
        self.feature_extractor = get_feature_extractor(backbone, return_nodes)
        feature_map_meta_data = dryrun_find_featuremap_dims(
            feature_extractor=self.feature_extractor,
            input_size=input_size,
            layers=return_nodes,
        )

        resolution = list(feature_map_meta_data.values())[0]["resolution"]
        if isinstance(resolution, int):
            self.scale = (resolution,) * 2
        elif isinstance(resolution, tuple):
            self.scale = resolution
        else:
            raise ValueError(
                f"Unknown type {type(resolution)} for `resolution`. Expected types are either int or tuple[int, int]."
            )
        self.descriptor = Descriptor(self.gamma_d, backbone)
        self.radius = torch.ones(1, requires_grad=True) * radius

        self.anomaly_map_generator = AnomalyMapGenerator(
            image_size=input_size, num_nearest_neighbors=num_nearest_neighbors
        )

    def compute_distance(self, target_oriented_features: Tensor) -> Tensor:
        if target_oriented_features.ndim == 4:
            target_oriented_features = rearrange(
                target_oriented_features, "b c h w -> b (h w) c")

        features = target_oriented_features.pow(2).sum(dim=2, keepdim=True)
        centers = self.memory_bank.pow(2).sum(
            dim=0, keepdim=True).to(features.device)
        f_c = 2 * torch.matmul(target_oriented_features,
                               (self.memory_bank.to(features.device)))
        distance = features + centers - f_c

        return distance

    def initialize_centroid(self, data_loader: DataLoader):
        device = next(self.feature_extractor.parameters()).device
        with torch.no_grad():
            for i, (x, _) in enumerate(tqdm(data_loader)):
                x = self.normalize(x)
                x = x.to(device, dtype=torch.float32)
                features = self.feature_extractor(x)
                features = list(features.values())
                target_features = self.descriptor(features)
                self.memory_bank = (
                    (self.memory_bank * i) + target_features.mean(dim=0, keepdim=True)) / (i + 1)

        self.memory_bank = rearrange(self.memory_bank, "b c h w -> (b h w) c")
        self.memory_bank = rearrange(self.memory_bank, "h w -> w h")

    def forward(self, input_tensor: Tensor) -> Tensor:
        input_tensor = self.normalize(input_tensor)
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            features = list(features.values())

        target_features = self.descriptor(features)
        distance = self.compute_distance(target_features)

        return distance if self.training else self.anomaly_map_generator(distance=distance, scale=self.scale)
