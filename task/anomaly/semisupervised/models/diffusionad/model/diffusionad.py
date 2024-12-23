from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor
import torchvision.transforms as transforms
from diffusers import UNet2DModel
from .module import DDPMScheduler, DiscriminativeSubNetwork


class DiffusionAD(nn.Module):
    def __init__(
        self, input_size: Tuple[int, int] = (256, 256),
        num_classes: int = 2, t_min: int = 100, t_max: int = 200,
        device: str = 'cpu'
    ):
        super(DiffusionAD, self).__init__()
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.t_max = t_max
        self.t_min = t_min
        self.device = device
        self.diffusion_scheduler = DDPMScheduler(
            num_train_timesteps=1000, beta_schedule="linear", prediction_type="epsilon"
        )
        self.denoising_subnet = UNet2DModel(
            sample_size=input_size,
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            block_out_channels=(
                128,
                256,
                256,
                512,
                512,
            ),
            attention_head_dim=4,
        )
        self.segment_subnet = DiscriminativeSubNetwork(
            in_channels=6, out_channels=num_classes
        )

    def forward(self, inputs: Tensor):
        # normalize
        inputs = self.normalize(inputs)

        # denoising
        noise = torch.randn(inputs.shape, dtype=(
            torch.float32), requires_grad=False).to(self.device)
        timesteps = torch.floor(torch.rand((inputs.shape[0],), requires_grad=False) * (
            self.t_max - self.t_min) + self.t_min).long().to(self.device)

        with torch.no_grad():
            noisy_images = self.diffusion_scheduler.add_noise(
                inputs, noise, timesteps)

        noise_pred = self.denoising_subnet(noisy_images, timesteps).sample

        # one step denoising
        with torch.no_grad():
            denoised_result = self.diffusion_scheduler.onestep_denoise(
                inputs, noise, timesteps, noise_pred
            ).to(self.device)

        joined_in = torch.cat([inputs, denoised_result], dim=1)
        prediction = self.segment_subnet(joined_in)

        if self.training:
            return noise, noise_pred, torch.softmax(prediction, dim=1)

        return 1 - torch.softmax(prediction, dim=1)[:, 0, ...]
