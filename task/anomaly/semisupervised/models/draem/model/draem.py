from typing import Dict, Callable
import torch
import torch.nn as nn
from torch import Tensor
import torchvision.transforms as transforms
from .module import ReconstructiveSubNetwork, DiscriminativeSubNetwork


class DRAEM(nn.Module):
    def __init__(self, num_classes: int = 2, sspcab: bool = False, sspcab_lambda: float = 0.1):
        super(DRAEM, self).__init__()
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.reconstructive_subnetwork = ReconstructiveSubNetwork(
            sspcab=sspcab)
        self.discriminative_subnetwork = DiscriminativeSubNetwork(
            in_channels=6, out_channels=num_classes)

        self.sspcab = sspcab

        if self.sspcab:
            self.sspcab_activations: Dict = {}
            self.setup_sspcab()
            self.sspcab_loss = nn.MSELoss()
            self.sspcab_lambda = sspcab_lambda

    def setup_sspcab(self):
        def get_activation(name: str) -> Callable:
            def hook(_, __, output: Tensor):
                self.sspcab_activations[name] = output

            return hook

        self.reconstructive_subnetwork.encoder.blocks[-1][-1].register_forward_hook(
            get_activation("input"))
        self.reconstructive_subnetwork.encoder.block5.register_forward_hook(
            get_activation("output"))

    def forward(self, batch: Tensor):
        # normalize
        batch = self.normalize(batch)

        # reconstruction
        reconstruction = self.reconstructive_subnetwork(batch)
        concatenated_inputs = torch.cat([batch, reconstruction], dim=1)

        # decoder
        prediction = self.discriminative_subnetwork(concatenated_inputs)

        if self.training:
            return reconstruction, torch.softmax(prediction, dim=1)

        return 1 - torch.softmax(prediction, dim=1)[:, 0, ...]
