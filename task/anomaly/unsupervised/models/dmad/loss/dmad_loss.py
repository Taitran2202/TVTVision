import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class DMADLoss(nn.Module):
    def __init__(self) -> None:
        super(DMADLoss, self).__init__()
        self.cos_loss = torch.nn.CosineSimilarity()

    def compute_layer_loss(self, fs_feats: Tensor, ft_feats: Tensor) -> Tensor:
        return torch.mean(1 - self.cos_loss(fs_feats.view(fs_feats.shape[0], -1), ft_feats.view(ft_feats.shape[0], -1)))

    def forward(self, fs_features: Tensor, ft_features: Tensor) -> Tensor:
        layer_losses: list[Tensor] = []
        for teacher_feature, ft_feature in zip(fs_features, ft_features):
            loss = self.compute_layer_loss(
                teacher_feature, ft_feature)
            layer_losses.append(loss)

        total_loss = torch.stack(layer_losses).sum()

        return total_loss
