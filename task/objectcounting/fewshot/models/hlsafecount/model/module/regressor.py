import torch.nn as nn


class Regressor(nn.Module):
    def __init__(self, in_dim):
        super(Regressor, self).__init__()
        self.regressor = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 1),
            nn.LeakyReLU(0.5, inplace=False),
            nn.Conv2d(in_dim // 2, in_dim // 4, 1),
            nn.LeakyReLU(0.5, inplace=False),
            nn.Conv2d(in_dim // 4, 1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.regressor(x)  # [bs,1,h,w]
