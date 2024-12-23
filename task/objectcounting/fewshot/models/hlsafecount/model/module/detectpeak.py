import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectPeaks(nn.Module):
    def __init__(self, threshold=0.1):
        super(DetectPeaks, self).__init__()
        self.threshold = threshold

    def scale(self, heatmap):
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap = torch.where(heatmap < self.threshold, torch.tensor(
            [0.0]).to(heatmap.device), heatmap)
        return heatmap

    def forward(self, heatmap):
        heatmap = self.scale(heatmap)
        local_max = F.max_pool2d(heatmap, kernel_size=(
            3, 3), stride=(1, 1), padding=0)
        local_max = F.pad(local_max, (1, 1, 0, 0), mode='reflect')
        local_max = F.pad(local_max, (0, 0, 1, 1))
        local_max = torch.eq(local_max, heatmap)
        background = torch.eq(heatmap, torch.tensor([0.0]).to(heatmap.device))
        eroded_background = 1.0 - background.float()
        eroded_background = F.max_pool2d(
            eroded_background, kernel_size=(3, 3), stride=(1, 1), padding=0)
        eroded_background = 1.0 - eroded_background
        eroded_background = F.pad(eroded_background, (1, 1, 1, 1), value=1)
        eroded_background = eroded_background.bool()
        detected_peaks = torch.logical_xor(local_max, eroded_background)
        return detected_peaks.int()
