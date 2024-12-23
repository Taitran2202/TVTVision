import math
from timm.models.vision_transformer import Block, PatchEmbed
import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import get_2d_sincos_pos_embed
from .conv_layernorm import Conv_LayerNorm


class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class MMR_Student(nn.Module):
    def __init__(self, input_size: tuple[int, int], patch_size: int, in_chans: int, embed_dim: int,
                 depth: int, num_heads: int, mlp_ratio: float, layers: list[str],
                 scale_factors: tuple[float, float], fpn_output_dim: tuple[int, int]):
        super(MMR_Student, self).__init__()
        self.input_size = torch.Size(input_size)
        self.layers = layers
        self.patch_embed = PatchEmbed(
            self.input_size[0], patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio,
                  qkv_bias=True, norm_layer=nn.LayerNorm)
            for i in range(depth)])

        self.norm = nn.LayerNorm(embed_dim)
        decoder_embed_dim = embed_dim
        self.decoder_FPN_mask_token = nn.Parameter(
            torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_FPN_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                                  requires_grad=False)
        strides = [int(patch_size / scale)
                   for scale in scale_factors]  # [4, 8, 16]

        self.stages = []
        use_bias = False
        for idx, scale in enumerate(scale_factors):
            out_dim = decoder_embed_dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(
                        decoder_embed_dim, decoder_embed_dim // 2, kernel_size=2, stride=2),
                    Conv_LayerNorm(decoder_embed_dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(
                        decoder_embed_dim // 2, decoder_embed_dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = decoder_embed_dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(
                    decoder_embed_dim, decoder_embed_dim // 2, kernel_size=2, stride=2)]
                out_dim = decoder_embed_dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(
                    f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    Conv2d(
                        out_dim,
                        fpn_output_dim[idx],
                        kernel_size=1,
                        bias=use_bias,
                        norm=Conv_LayerNorm(fpn_output_dim[idx]),
                    ),
                    Conv2d(
                        fpn_output_dim[idx],
                        fpn_output_dim[idx],
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=Conv_LayerNorm(fpn_output_dim[idx]),
                    ),
                ]
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_FPN_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_FPN_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.decoder_FPN_mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, imgs, mask_ratio: float = 0.75, ids_shuffle=None):
        latent, _, ids_restore = self.forward_encoder(
            imgs, mask_ratio, ids_shuffle=ids_shuffle)
        reverse_features = self.forward_decoder_FPN(
            latent, ids_restore)  # [N, L, p*p*3]
        return reverse_features

    def forward_encoder(self, x, mask_ratio, ids_shuffle=None):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(
            x, mask_ratio, ids_shuffle=ids_shuffle)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def random_masking(self, x, mask_ratio, ids_shuffle=None):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        if ids_shuffle is None:
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
            ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_decoder_FPN(self, x, ids_restore):
        mask_tokens = self.decoder_FPN_mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x_ = x_ + self.decoder_FPN_pos_embed[:, 1:, :]

        # FPN stage
        h = w = int(x_.shape[1] ** 0.5)
        decoder_dim = x_.shape[2]

        x = x_.permute(0, 2, 1).view(-1, decoder_dim,
                                     h, w)  # (B, channel, h, w)
        results = []

        for _, stage in enumerate(self.stages):
            stage_feature_map = stage(x)
            results.append(stage_feature_map)

        return {layer: feature for layer, feature in zip(self.layers, results)}
