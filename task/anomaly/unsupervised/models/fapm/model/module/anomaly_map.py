from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyMapGenerator(nn.Module):
    def __init__(self, input_size) -> None:
        super().__init__()
        self.input_size = input_size if isinstance(
            input_size, tuple) else (input_size, input_size)

    def compute_anomaly_map(self, patch_scores: torch.Tensor, feature_map_shape: torch.Size) -> torch.Tensor:
        """Pixel Level Anomaly Heatmap.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores
            feature_map_shape (torch.Size): 2-D feature map shape (width, height)

        Returns:
            torch.Tensor: Map of the pixel-level anomaly scores
        """
        width, height = feature_map_shape
        batch_size = len(patch_scores) // (width * height)

        anomaly_map = patch_scores[:, 0].reshape(
            (batch_size, 1, width, height))
        anomaly_map = F.interpolate(anomaly_map, size=(
            self.input_size[0], self.input_size[1]))

        return anomaly_map

    def compute_anomaly_score(self, patch_scores: torch.Tensor) -> torch.Tensor:
        """Compute Image-Level Anomaly Score.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores
        Returns:
            torch.Tensor: Image-level anomaly scores
        """
        max_scores, _ = torch.max(patch_scores[:, 0], dim=0)
        max_scores = max_scores.unsqueeze(0)
        confidence = torch.index_select(
            patch_scores, 0, torch.argmax(patch_scores[:, 0], dim=0))
        confidence = confidence.unsqueeze(0)
        weights = 1 - (torch.exp(confidence).max() /
                       torch.exp(confidence).sum())
        weights = weights.unsqueeze(0)
        score = weights * max_scores

        return score

    def forward(self, **kwargs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        patch_scores = kwargs["patch_scores"]
        feature_map_shape = kwargs["feature_map_shape"]

        anomaly_map = self.compute_anomaly_map(patch_scores, feature_map_shape)
        anomaly_score = self.compute_anomaly_score(patch_scores)
        return anomaly_map, anomaly_score
