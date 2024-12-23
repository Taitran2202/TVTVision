import torch
import torch.nn as nn
import torch.nn.functional as F


class BDCNLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BDCNLoss, self).__init__()

    def forward(self, inputs, targets, l_weight=1.1):
        # bdcn loss modified in DexiNed
        label = targets.float()
        prediction = inputs.float()
        mask = targets.clone()
        num_positive = torch.sum(label == 1.).float()
        num_negative = torch.sum(label == 0.).float()

        mask[label == 1.] = 1.0 * num_negative / (num_positive + num_negative)
        mask[label == 0.] = 1.1 * num_positive / (num_positive + num_negative)
        mask[label == 2.] = 0
        # cost = nn.BCELoss(mask, reduction='none')(
        #     prediction, targets.float())
        # cost = torch.sum(cost.float().mean((1, 2, 3)))  # before sum
        cost = F.binary_cross_entropy(
            prediction, targets, weight=mask, reduction='sum')
        return l_weight * cost


def bdrloss(prediction, label, radius, device):
    filt = torch.ones(1, 1, 2*radius+1, 2*radius+1)
    filt.requires_grad = False
    filt = filt.to(device)

    bdr_pred = prediction * label
    pred_bdr_sum = label * \
        F.conv2d(bdr_pred, filt, bias=None, stride=1, padding=radius)

    texture_mask = F.conv2d(label.float(), filt,
                            bias=None, stride=1, padding=radius)
    mask = (texture_mask != 0).float()
    mask[label == 1] = 0
    pred_texture_sum = F.conv2d(
        prediction * (1-label) * mask, filt, bias=None, stride=1, padding=radius)

    softmax_map = torch.clamp(
        pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10), 1e-10, 1 - 1e-10)
    cost = -label * torch.log(softmax_map)
    cost[label == 0] = 0

    return cost.sum()
    # return torch.sum(cost.float().mean((1, 2, 3)))


def textureloss(prediction, label, mask_radius, device):
    filt1 = torch.ones(1, 1, 3, 3)
    filt1.requires_grad = False
    filt1 = filt1.to(device)
    filt2 = torch.ones(1, 1, 2*mask_radius+1, 2*mask_radius+1)
    filt2.requires_grad = False
    filt2 = filt2.to(device)

    pred_sums = F.conv2d(prediction.float(), filt1,
                         bias=None, stride=1, padding=1)
    label_sums = F.conv2d(label.float(), filt2, bias=None,
                          stride=1, padding=mask_radius)

    mask = 1 - torch.gt(label_sums, 0).float()

    loss = -torch.log(torch.clamp(1-pred_sums/9, 1e-10, 1-1e-10))
    loss[mask == 0] = 0

    return torch.sum(loss)
    # return torch.sum(loss.float().mean((1, 2, 3)))


class RCFLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(RCFLoss, self).__init__()

    def forward(self, inputs, targets, l_weight=[0., 0.], device="cpu"):
        tex_factor, bdr_factor = l_weight

        label = targets.float()
        prediction = inputs.float()
        mask = targets.clone()

        num_positive = torch.sum(label == 1.).float()
        num_negative = torch.sum(label == 0.).float()

        mask[label == 1.] = 1.0 * num_negative / (num_positive + num_negative)
        mask[label == 0.] = 1.1 * num_positive / (num_positive + num_negative)
        mask[label == 2.] = 0

        cost = F.binary_cross_entropy(
            prediction.float(), label.float(), weight=mask, reduction='sum')
        # cost = torch.sum(cost.float().mean((1, 2, 3)))
        label_w = (label != 0).float()
        textcost = textureloss(
            prediction.float(), label_w.float(), mask_radius=4, device=device)
        bdrcost = bdrloss(prediction.float(), label_w.float(),
                          radius=4, device=device)

        return cost + bdr_factor * bdrcost + tex_factor * textcost
