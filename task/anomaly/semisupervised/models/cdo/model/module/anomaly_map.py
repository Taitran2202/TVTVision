import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class AnomalyMapGenerator(nn.Module):
    def __init__(self, input_size) -> None:
        super(AnomalyMapGenerator, self).__init__()
        self.input_size = input_size if isinstance(
            input_size, tuple) else tuple(input_size)

    def compute_layer_map(self, teacher_features: Tensor, student_features: Tensor) -> Tensor:
        norm_teacher_features = F.normalize(teacher_features)
        norm_student_features = F.normalize(student_features)

        layer_map = 0.5 * torch.norm(norm_teacher_features -
                                     norm_student_features, p=2, dim=-3, keepdim=True) ** 2
        layer_map = F.interpolate(
            layer_map, size=self.input_size, align_corners=False, mode="bilinear")
        return layer_map

    def compute_anomaly_map(
            self, teacher_features: Tensor, student_features: Tensor) -> torch.Tensor:
        batch_size = teacher_features[0].shape[0]
        anomaly_map = torch.ones(
            batch_size, 1, self.input_size[0], self.input_size[1])
        for teacher_feature, student_feature in zip(teacher_features, student_features):
            layer_map = self.compute_layer_map(
                teacher_feature, student_feature)
            anomaly_map = anomaly_map.to(layer_map.device)
            anomaly_map += layer_map

        return anomaly_map

    def forward(self, teacher_features: Tensor, student_features: Tensor) -> torch.Tensor:
        return self.compute_anomaly_map(teacher_features, student_features)
