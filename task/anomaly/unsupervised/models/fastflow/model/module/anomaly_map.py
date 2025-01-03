import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap."""

    def __init__(self, input_size) -> None:
        super().__init__()
        self.input_size = input_size if isinstance(
            input_size, tuple) else tuple(input_size)

    def forward(self, hidden_variables) -> Tensor:
        flow_maps = []

        for hidden_variable in hidden_variables:
            log_prob = -torch.mean(hidden_variable**2,
                                   dim=1, keepdim=True) * 0.5
            prob = torch.exp(log_prob)
            flow_map = F.interpolate(
                input=-prob,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )
            flow_maps.append(flow_map)

        flow_maps = torch.stack(flow_maps, dim=-1)
        anomaly_map = torch.mean(flow_maps, dim=-1)
        return anomaly_map[:, 0, ...]
