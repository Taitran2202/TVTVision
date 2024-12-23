import torch
import torch.nn as nn


def get_dice_loss(gt_score, pred_score, mask, weights=None):
    if weights is not None:
        mask = weights * mask
    inter = torch.sum(gt_score * pred_score * mask)
    union = torch.sum(gt_score * mask) + torch.sum(pred_score * mask) + 1e-6
    return 1. - (2 * inter / union)


class EASTv2Loss(nn.Module):
    def __init__(self, weight=0.01):
        super(EASTv2Loss, self).__init__()
        self.weight = weight

    def forward(self, gt_score, pred_score, gt_geo, pred_geo, training_mask):
        dice_loss = get_dice_loss(gt_score, pred_score, training_mask)

        # smooth l1 loss
        gt_geo_split = torch.chunk(gt_geo, chunks=8 + 1, dim=1)
        pred_geo_split = torch.chunk(pred_geo, chunks=8, dim=1)
        smooth_l1 = 0
        for i in range(0, 8):
            geo_diff = gt_geo_split[i] - pred_geo_split[i]
            abs_geo_diff = torch.abs(geo_diff)
            smooth_l1_sign = torch.le(abs_geo_diff, gt_score)
            smooth_l1_sign = smooth_l1_sign.type(torch.float32)
            in_loss = abs_geo_diff * abs_geo_diff * smooth_l1_sign + \
                (abs_geo_diff - 0.5) * (1.0 - smooth_l1_sign)
            out_loss = gt_geo_split[-1] / 8 * in_loss * gt_score
            smooth_l1 += out_loss
        smooth_l1_loss = torch.mean(smooth_l1 * gt_score)

        losses = dice_loss * self.weight + smooth_l1_loss
        return losses
