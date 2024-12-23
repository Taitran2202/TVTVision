import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, num_classes: int = 2):
        super(Decoder, self).__init__()

        def layer(in_channels, out_channels):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def double_layer(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def shorcut_layer(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.up1 = layer(512, 256)
        self.db1 = double_layer(256 + 256, 256)
        self.db1_shor_cut = shorcut_layer(256 + 256, 256)

        self.up2 = layer(256, 128)
        self.db2 = double_layer(128 + 128, 128)
        self.db2_shor_cut = shorcut_layer(128 + 128, 128)

        self.up3 = layer(128, 64)
        self.db3 = double_layer(64 + 64, 64)
        self.db3_shor_cut = shorcut_layer(64 + 64, 64)

        self.up4 = layer(64, 32)
        self.db4 = double_layer(32 + 64, 48)
        self.db4_shor_cut = shorcut_layer(32 + 64, 48)

        self.up5 = layer(48, 48)
        self.db5 = double_layer(48, 24)

        self.final_out = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, num_classes, kernel_size=3, padding=1),
        )

        for m in [self.up1, self.up2, self.up3, self.up4, self.up5,
                  self.db1, self.db2, self.db3, self.db4, self.db5,
                  self.db1_shor_cut, self.db2_shor_cut, self.db3_shor_cut, self.db4_shor_cut, self.final_out]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, encoder_output, concat_features):
        # concat_features = [level0, level1, level2, level3]
        f0, f1, f2, f3 = concat_features

        # 512 x 8 x 8 -> 512 x 16 x 16
        up1 = self.up1(encoder_output)
        cat = torch.cat((up1, f3), dim=1)
        db1 = self.db1(cat)
        db1 = db1 + self.db1_shor_cut(cat)

        # 512 x 16 x 16 -> 256 x 32 x 32
        up2 = self.up2(db1)
        cat = torch.cat((up2, f2), dim=1)
        db2 = self.db2(cat)
        db2 = db2 + self.db2_shor_cut(cat)

        # 256 x 32 x 32 -> 128 x 64 x 64
        up3 = self.up3(db2)
        cat = torch.cat((up3, f1), dim=1)
        db3 = self.db3(cat)
        db3 = db3 + self.db3_shor_cut(cat)

        # 128 x 64 x 64 -> 96 x 128 x 128
        up4 = self.up4(db3)
        cat = torch.cat((up4, f0), dim=1)
        db4 = self.db4(cat)
        db4 = db4 + self.db4_shor_cut(cat)

        # 96 x 128 x 128 -> 48 x 256 x 256
        up5 = self.up5(db4)

        # 48 x 256 x 256 -> 2 x 256 x 256
        db5 = self.db5(up5)

        out = self.final_out(db5)

        return out
