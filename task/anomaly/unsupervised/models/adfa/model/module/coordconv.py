import torch
import torch.nn as nn
from torch import Tensor


class AddCoords(nn.Module):
    def __init__(self, with_r: bool = False) -> None:
        super(AddCoords, self).__init__()
        self.with_r = with_r

    def forward(self, input_tensor: Tensor) -> Tensor:
        batch, _, x_dim, y_dim = input_tensor.shape
        xx_ones = torch.ones([1, 1, 1, y_dim], dtype=torch.int32)
        yy_ones = torch.ones([1, 1, 1, x_dim], dtype=torch.int32)

        xx_range = torch.arange(x_dim, dtype=torch.int32)
        yy_range = torch.arange(y_dim, dtype=torch.int32)
        xx_range = xx_range[None, None, :, None]
        yy_range = yy_range[None, None, :, None]

        xx_channel = torch.matmul(xx_range, xx_ones)
        yy_channel = torch.matmul(yy_range, yy_ones)

        # Transpose y
        yy_channel = yy_channel.permute(0, 1, 3, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch, 1, 1, 1).to(input_tensor.device)
        yy_channel = yy_channel.repeat(batch, 1, 1, 1).to(input_tensor.device)

        out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr_channel = torch.sqrt(
                torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            out = torch.cat([out, rr_channel], dim=1)

        return out


class CoordConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        # AddCoord layer.
        self.add_coords = AddCoords(with_r)

        # Create conv layer on top of add_coords layer.
        self.conv2d = nn.Conv2d(
            # 2 for rank-2 tensor, 1 for r if with_r
            in_channels=in_channels + 2 + int(with_r),
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input_tensor: Tensor) -> Tensor:  # pylint: disable=arguments-renamed
        """Forward pass.
        Args:
            input_tensor (Tensor): Input tensor.
        Returns:
            Tensor: Output tensor after applying the CoordConv layer.
        """
        out = self.add_coords(input_tensor)
        out = self.conv2d(out)
        return out
