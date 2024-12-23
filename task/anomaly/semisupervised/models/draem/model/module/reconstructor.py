import torch
import torch.nn as nn
from torch import Tensor
from .sspcab import SSPCAB


class EncoderReconstructive(nn.Module):
    def __init__(self, in_channels: int, base_width: int, sspcab: bool = False):
        super(EncoderReconstructive, self).__init__()
        self.in_channels = in_channels
        self.base_width = base_width
        self.sspcab = sspcab
        self.num_blocks = 4
        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            channels = self.base_width * (2 ** i)
            block = nn.Sequential(
                nn.Conv2d(self.in_channels, channels,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
            self.blocks.append(block)
            self.in_channels = channels
        if sspcab:
            self.block5 = SSPCAB(base_width * 8)
        else:
            self.block5 = nn.Sequential(
                nn.Conv2d(base_width * 8, base_width *
                          8, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_width * 8),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_width * 8, base_width *
                          8, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_width * 8),
                nn.ReLU(inplace=True),
            )

    def forward(self, batch: Tensor) -> Tensor:
        for _, block in enumerate(self.blocks):
            batch = block(batch)
        return self.block5(batch)


class DecoderReconstructive(nn.Module):
    def __init__(self, base_width: int, out_channels: int = 1):
        super(DecoderReconstructive, self).__init__()

        base_width = base_width * 8

        up_layers = []
        db_layers = []
        for i in range(4):
            up_layers.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear",
                            align_corners=True),
                nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_width),
                nn.ReLU(inplace=True),
            ))
            db_layers.append(nn.Sequential(
                nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_width, base_width // 2,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(base_width // 2),
                nn.ReLU(inplace=True),
            ))
            base_width = base_width // 2

        self.up_layers = nn.ModuleList(up_layers)
        self.db_layers = nn.ModuleList(db_layers)

        self.fin_out = nn.Sequential(
            nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, act5: Tensor) -> Tensor:
        x = act5
        for i in range(4):
            x = self.up_layers[i](x)
            x = self.db_layers[i](x)

        return self.fin_out(x)


class ReconstructiveSubNetwork(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_width=128, sspcab: bool = False):
        super(ReconstructiveSubNetwork, self).__init__()
        self.encoder = EncoderReconstructive(
            in_channels, base_width, sspcab=sspcab)
        self.decoder = DecoderReconstructive(
            base_width, out_channels=out_channels)

        self.Init()

    def Init(self):
        mo_list = [self.encoder, self.decoder]
        for m in mo_list:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, batch: Tensor) -> Tensor:
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)

        return decoded
