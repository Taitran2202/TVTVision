from typing import List
import torch
import torch.nn as nn


class DiffModule(nn.Module):
    def __init__(self):
        super(DiffModule, self).__init__()

    def forward(self, features_normal: List[torch.Tensor], features: List[torch.Tensor]) -> torch.Tensor:
        for level in range(len(features)):
            diff_features = (features_normal[level] - features[level]) ** 2
            features[level] = torch.cat(
                [features[level], diff_features], dim=1)

        return features
