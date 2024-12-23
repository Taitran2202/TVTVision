import torch
import torch.nn as nn
from torch import Tensor


class ADFALoss(nn.Module):
    def __init__(self, num_nearest_neighbors: int, num_hard_negative_features: int, radius: float) -> None:
        super(ADFALoss, self).__init__()
        self.num_nearest_neighbors = num_nearest_neighbors
        self.num_hard_negative_features = num_hard_negative_features
        self.radius = torch.ones(1, requires_grad=True) * radius

    def forward(self, distance: Tensor) -> Tensor:
        num_neighbors = self.num_nearest_neighbors + self.num_hard_negative_features
        distance = distance.topk(num_neighbors, largest=False).values

        score = distance[:, :, : self.num_nearest_neighbors] - \
            (self.radius**2).to(distance.device)
        l_att = torch.mean(torch.max(torch.zeros_like(score), score))

        score = (self.radius**2).to(distance.device) - \
            distance[:, :, self.num_hard_negative_features:]
        l_rep = torch.mean(torch.max(torch.zeros_like(score), score - 0.1))

        loss = (l_att + l_rep) * 1000

        return loss
