import torch
import torch.nn as nn
from torch import Tensor


class MMRLoss(nn.Module):
    def __init__(self) -> None:
        super(MMRLoss, self).__init__()
        self.cos_loss = nn.CosineSimilarity()

    def compute_layer_loss(self, teacher_feats: Tensor, student_feats: Tensor) -> Tensor:
        teacher_feats = teacher_feats.permute(0, 2, 3, 1)
        student_feats = student_feats.permute(0, 2, 3, 1)
        layer_loss = torch.mean(1 - self.cos_loss(teacher_feats.contiguous().view(-1, teacher_feats.shape[-1]),
                                                  student_feats.contiguous().view(-1, student_feats.shape[-1])))

        return layer_loss

    def forward(self, teacher_features: dict[str, Tensor], student_features: dict[str, Tensor]) -> Tensor:
        layer_losses: list[Tensor] = []
        for layer in teacher_features.keys():
            loss = self.compute_layer_loss(
                teacher_features[layer], student_features[layer])
            layer_losses.append(loss)

        total_loss = torch.stack(layer_losses).sum()

        return total_loss
