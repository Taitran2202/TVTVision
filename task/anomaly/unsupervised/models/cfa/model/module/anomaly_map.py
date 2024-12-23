from einops import rearrange
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class AnomalyMapGenerator(nn.Module):
    def __init__(self, image_size: tuple, num_nearest_neighbors: int) -> None:
        super(AnomalyMapGenerator, self).__init__()
        self.image_size = image_size if isinstance(
            image_size, tuple) else tuple(image_size)
        self.num_nearest_neighbors = num_nearest_neighbors

    def compute_score(self, distance: Tensor, scale: tuple[int, int]) -> Tensor:
        distance = torch.sqrt(distance)
        distance = distance.topk(
            self.num_nearest_neighbors, largest=False).values
        distance = (F.softmin(distance, dim=-1)[:, :, 0]) * distance[:, :, 0]
        distance = distance.unsqueeze(-1)

        score = rearrange(distance, "b (h w) c -> b c h w",
                          h=scale[0], w=scale[1])
        return score.detach()

    def compute_anomaly_map(self, score: Tensor) -> Tensor:
        anomaly_map = score.mean(dim=1, keepdim=True)
        anomaly_map = F.interpolate(
            anomaly_map, size=self.image_size, mode="bilinear", align_corners=False)

        return anomaly_map[:, 0, ...]

    def forward(self, distance, scale) -> Tensor:
        score = self.compute_score(distance=distance, scale=scale)
        anomaly_map = self.compute_anomaly_map(score)

        return anomaly_map
