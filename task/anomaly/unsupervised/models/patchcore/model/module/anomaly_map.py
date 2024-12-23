import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from utils.filters import GaussianBlur2d


class AnomalyMapGenerator(nn.Module):
    def __init__(self, input_size, sigma: int = 4) -> None:
        super().__init__()
        self.input_size = input_size
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(kernel_size=(
            kernel_size, kernel_size), sigma=(sigma, sigma), channels=1)

    def compute_anomaly_map(self, patch_scores: Tensor) -> Tensor:
        anomaly_map = F.interpolate(patch_scores, size=(
            self.input_size[0], self.input_size[1]))

        anomaly_map = self.blur(anomaly_map)

        return anomaly_map

    def forward(self, patch_scores: Tensor) -> Tensor:
        anomaly_map = self.compute_anomaly_map(patch_scores)
        return anomaly_map
