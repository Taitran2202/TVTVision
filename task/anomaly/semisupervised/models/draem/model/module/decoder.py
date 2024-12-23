from typing import Tuple
import torch
import torch.nn as nn
from torch import Tensor


class EncoderDiscriminative(nn.Module):
    def __init__(self, in_channels: int, base_width: int):
        super(EncoderDiscriminative, self).__init__()

        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.block1 = block(in_channels, base_width)
        self.mp1 = nn.MaxPool2d(2)
        self.block2 = block(base_width, base_width * 2)
        self.mp2 = nn.MaxPool2d(2)
        self.block3 = block(base_width * 2, base_width * 4)
        self.mp3 = nn.MaxPool2d(2)
        self.block4 = block(base_width * 4, base_width * 8)
        self.mp4 = nn.MaxPool2d(2)
        self.block5 = block(base_width * 8, base_width * 8)
        self.mp5 = nn.MaxPool2d(2)
        self.block6 = block(base_width * 8, base_width * 8)

    def forward(self, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        act1 = self.block1(batch)
        mp1 = self.mp1(act1)
        act2 = self.block2(mp1)
        mp2 = self.mp2(act2)
        act3 = self.block3(mp2)
        mp3 = self.mp3(act3)
        act4 = self.block4(mp3)
        mp4 = self.mp4(act4)
        act5 = self.block5(mp4)
        mp5 = self.mp5(act5)
        act6 = self.block6(mp5)

        return act1, act2, act3, act4, act5, act6


class DecoderDiscriminative(nn.Module):
    def __init__(self, base_width: int, out_channels: int = 1):
        super(DecoderDiscriminative, self).__init__()

        def block(in_channels, out_channels, upsample=False):
            layers = []
            if upsample:
                layers.append(nn.Upsample(scale_factor=2,
                              mode="bilinear", align_corners=True))
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                       nn.BatchNorm2d(out_channels),
                       nn.ReLU(inplace=True)]
            if not upsample:
                layers.append(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.up_b = block(base_width * 8, base_width * 8, True)
        self.db_b = block(base_width * (8 + 8), base_width * 8)

        self.up1 = block(base_width * 8, base_width * 4, True)
        self.db1 = block(base_width * (4 + 8), base_width * 4)

        self.up2 = block(base_width * 4, base_width * 2, True)
        self.db2 = block(base_width * (2 + 4), base_width * 2)

        self.up3 = block(base_width * 2, base_width, True)
        self.db3 = block(base_width * (2 + 1), base_width)

        self.up4 = block(base_width, base_width, True)
        self.db4 = block(base_width * 2, base_width)

        self.fin_out = nn.Sequential(
            nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, act1: Tensor, act2: Tensor, act3: Tensor, act4: Tensor, act5: Tensor, act6: Tensor) -> Tensor:
        up_b = self.up_b(act6)
        cat_b = torch.cat((up_b, act5), dim=1)
        db_b = self.db_b(cat_b)

        up1 = self.up1(db_b)
        cat1 = torch.cat((up1, act4), dim=1)
        db1 = self.db1(cat1)

        up2 = self.up2(db1)
        cat2 = torch.cat((up2, act3), dim=1)
        db2 = self.db2(cat2)

        up3 = self.up3(db2)
        cat3 = torch.cat((up3, act2), dim=1)
        db3 = self.db3(cat3)

        up4 = self.up4(db3)
        cat4 = torch.cat((up4, act1), dim=1)
        db4 = self.db4(cat4)

        return self.fin_out(db4)


class DiscriminativeSubNetwork(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_width: int = 64):
        super(DiscriminativeSubNetwork, self).__init__()
        self.encoder_segment = EncoderDiscriminative(in_channels, base_width)
        self.decoder_segment = DecoderDiscriminative(
            base_width, out_channels=out_channels)

        self.Init()

    def Init(self):
        mo_list = [self.encoder_segment, self.decoder_segment]
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
        act1, act2, act3, act4, act5, act6 = self.encoder_segment(batch)
        segmentation = self.decoder_segment(act1, act2, act3, act4, act5, act6)

        return segmentation
