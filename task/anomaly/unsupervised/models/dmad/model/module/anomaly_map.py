import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from utils.filters import GaussianBlur2d


class AnomalyMapGenerator(nn.Module):
    def __init__(self, input_size, sigma: int = 4, amap_mode='add') -> None:
        super(AnomalyMapGenerator, self).__init__()
        self.input_size = input_size if isinstance(
            input_size, tuple) else tuple(input_size)
        self.amap_mode = amap_mode
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(kernel_size=(
            kernel_size, kernel_size), sigma=(sigma, sigma), channels=1)

    def compute_layer_map(self, en_features: Tensor, de_features: Tensor) -> Tensor:
        layer_map = 1 - F.cosine_similarity(en_features, de_features)
        layer_map = torch.unsqueeze(layer_map, dim=1)
        layer_map = F.interpolate(
            layer_map, size=self.input_size, align_corners=True, mode="bilinear")
        return layer_map

    def compute_anomaly_map(
        self, en_features: Tensor, de_features: Tensor, grid1: Tensor, grid2: Tensor
    ) -> torch.Tensor:
        batch_size = en_features[0].shape[0]
        if self.amap_mode == 'mul':
            anomaly_map = torch.ones(
                batch_size, 1, self.input_size[0], self.input_size[1])
        else:
            anomaly_map = torch.zeros(
                batch_size, 1, self.input_size[0], self.input_size[1])
        for layer in range(len(en_features)):
            layer_map = self.compute_layer_map(
                en_features[layer], de_features[layer])
            anomaly_map = anomaly_map.to(layer_map.device)

            if self.amap_mode == 'mul':
                anomaly_map *= layer_map
            else:
                anomaly_map += layer_map

        anomaly_map = F.grid_sample(F.grid_sample(
            anomaly_map, grid2, align_corners=True), grid1, align_corners=True)
        anomaly_map = self.blur(anomaly_map)

        return anomaly_map

    def forward(self, en_features: Tensor, de_features: Tensor, grid1: Tensor, grid2: Tensor) -> torch.Tensor:
        return self.compute_anomaly_map(en_features, de_features, grid1, grid2)
