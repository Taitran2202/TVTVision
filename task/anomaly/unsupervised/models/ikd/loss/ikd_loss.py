import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class IKDLoss(nn.Module):
    def __init__(self) -> None:
        super(IKDLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def compute_layer_loss(self, teacher_feats: Tensor, student_feats: Tensor) -> Tensor:
        height, width = teacher_feats.shape[2:]

        norm_teacher_features = F.normalize(teacher_feats)
        norm_student_features = F.normalize(student_feats)
        layer_loss = (0.5 / (width * height)) * \
            self.mse_loss(norm_teacher_features, norm_student_features)

        return layer_loss

    def forward(self, teacher_features: Tensor, student_features: Tensor) -> Tensor:
        layer_losses: list[Tensor] = []
        for teacher_feature, student_feature in zip(teacher_features, student_features):
            loss = self.compute_layer_loss(
                teacher_feature, student_feature)
            layer_losses.append(loss)

        total_loss = torch.stack(layer_losses).sum()

        return total_loss
