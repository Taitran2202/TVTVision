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
        self, teacher_features: dict[str, Tensor], student_features: dict[str, Tensor]
    ) -> torch.Tensor:
        batch_size = list(teacher_features.values())[0].shape[0]
        anomaly_map = torch.ones(
            batch_size, 1, self.input_size[0], self.input_size[1])
        for layer in teacher_features.keys():
            layer_map = self.compute_layer_map(
                teacher_features[layer], student_features[layer])
            anomaly_map = anomaly_map.to(layer_map.device)
            anomaly_map *= layer_map

        return anomaly_map

    def forward(self, teacher_features: dict[str, Tensor], student_features: dict[str, Tensor]) -> torch.Tensor:
        return self.compute_anomaly_map(teacher_features, student_features)
