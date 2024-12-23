from typing import List, cast
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap."""

    def __init__(
        self,
        image_size: tuple,
        pool_layers: list[str],
    ) -> None:
        super().__init__()
        self.distance = torch.nn.PairwiseDistance(p=2, keepdim=True)
        self.image_size = image_size if isinstance(
            image_size, tuple) else tuple(image_size)
        self.pool_layers: list[str] = pool_layers

    def compute_anomaly_map(self, distribution: list[Tensor], height: list[int], width: list[int]) -> Tensor:
        layer_maps: list[Tensor] = []
        for layer_idx in range(len(self.pool_layers)):
            layer_distribution = distribution[layer_idx].clone().detach()
            # Normalize the likelihoods to (-Inf:0] and convert to probs in range [0:1]
            layer_probabilities = torch.exp(
                layer_distribution - layer_distribution.max())
            layer_map = layer_probabilities.reshape(
                -1, height[layer_idx], width[layer_idx])
            # upsample
            layer_maps.append(
                F.interpolate(
                    layer_map.unsqueeze(1), size=self.image_size, mode="bilinear", align_corners=True
                ).squeeze(1)
            )
        # score aggregation
        score_map = torch.zeros_like(layer_maps[0])
        for layer_idx in range(len(self.pool_layers)):
            score_map += layer_maps[layer_idx]

        # Invert probs to anomaly scores
        anomaly_map = score_map.max() - score_map

        return anomaly_map

    def forward(self, **kwargs: list[Tensor] | list[int] | list[list]) -> Tensor:
        distribution: list[Tensor] = cast(List[Tensor], kwargs["distribution"])
        height: list[int] = cast(List[int], kwargs["height"])
        width: list[int] = cast(List[int], kwargs["width"])

        return self.compute_anomaly_map(distribution, height, width)
