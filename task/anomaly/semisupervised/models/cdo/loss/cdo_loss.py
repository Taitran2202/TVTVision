import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class CDOLoss(nn.Module):
    def __init__(self, gamma: int, OOM: bool, aggregation: bool) -> None:
        super(CDOLoss, self).__init__()
        self.gamma = gamma
        self.OOM = OOM
        self.aggregation = aggregation

    def cal_discrepancy(self, fe, fa, OOM, normal, gamma, aggregation=True):
        fe = F.normalize(fe, p=2, dim=1)
        fa = F.normalize(fa, p=2, dim=1)
        d_p = torch.sum((fe - fa) ** 2, dim=1)
        if OOM:
            mu_p = torch.mean(d_p)
            if normal:
                w = (d_p / mu_p) ** gamma
            else:
                w = (mu_p / d_p) ** gamma
            w = w.detach()
        else:
            w = torch.ones_like(d_p)
        if aggregation:
            d_p = torch.sum(d_p * w)
        sum_w = torch.sum(w)

        return d_p, sum_w

    def forward(self, teacher_features: Tensor, student_features: Tensor, mask=None) -> Tensor:
        loss = 0
        _, _, H_0, W_0 = teacher_features[0].shape
        for i in range(len(teacher_features)):
            teacher_features[i] = F.interpolate(teacher_features[i], size=(
                H_0, W_0), mode='bilinear', align_corners=True)
            student_features[i] = F.interpolate(student_features[i], size=(
                H_0, W_0), mode='bilinear', align_corners=True)

        for fe, fa in zip(teacher_features, student_features):
            B, C, H, W = fe.shape

            if mask is not None:
                mask_vec = F.interpolate(
                    mask.unsqueeze(1), (H, W), mode='nearest')
            else:
                mask_vec = torch.zeros((B, C, H, W))

            mask_vec = mask_vec.permute(0, 2, 3, 1).reshape(-1, )
            fe = fe.permute(0, 2, 3, 1).reshape(-1, C)
            fa = fa.permute(0, 2, 3, 1).reshape(-1, C)
            fe_n = fe[mask_vec == 0].reshape(-1, C)
            fa_n = fa[mask_vec == 0].reshape(-1, C)
            fe_s = fe[mask_vec != 0].reshape(-1, C)
            fa_s = fa[mask_vec != 0].reshape(-1, C)

            loss_n, weight_n = self.cal_discrepancy(fe_n, fa_n, OOM=self.OOM, normal=True, gamma=self.gamma,
                                                    aggregation=self.aggregation)
            loss_s, weight_s = self.cal_discrepancy(fe_s, fa_s, OOM=self.OOM, normal=False, gamma=self.gamma,
                                                    aggregation=self.aggregation)

            loss += ((loss_n - loss_s) / (weight_n + weight_s) * B)

        return loss
