import timm
from timm.models.cait import Cait
from timm.models.vision_transformer import VisionTransformer
from FrEIA.framework import SequenceINN
from FrEIA.modules import AllInOneBlock
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from .module import AnomalyMapGenerator


def subnet_conv_func(kernel_size: int, hidden_ratio: float):

    def subnet_conv(in_channels: int, out_channels: int):
        hidden_channels = int(in_channels * hidden_ratio)
        padding = 2 * (kernel_size // 2 - ((1 + kernel_size) %
                       2), kernel_size // 2)
        return nn.Sequential(
            nn.ZeroPad2d(padding),
            nn.Conv2d(in_channels, hidden_channels, kernel_size),
            nn.ReLU(),
            nn.ZeroPad2d(padding),
            nn.Conv2d(hidden_channels, out_channels, kernel_size),
        )

    return subnet_conv


def create_fast_flow_block(input_dimensions, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    nodes = SequenceINN(*input_dimensions)

    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        nodes.append(
            AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )

    return nodes


class FastFlow(nn.Module):
    def __init__(
        self,
        input_size,
        feature_extractor_name: str,
        flow_steps: int = 8,
        conv3x3_only: bool = False,
        hidden_ratio: float = 1.0,
    ) -> None:
        super(FastFlow, self).__init__()

        self.input_size = input_size
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        if feature_extractor_name in ("cait_m48_448", "deit_base_distilled_patch16_384"):
            self.feature_extractor = timm.create_model(
                feature_extractor_name, pretrained=True)
            channels = [768]
            scales = [16]

        elif feature_extractor_name in ("resnet18", "wide_resnet50_2"):
            self.feature_extractor = timm.create_model(
                feature_extractor_name,
                pretrained=True,
                features_only=True,
                out_indices=[1, 2, 3],
            )
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()

            # for transformers, use their pretrained norm w/o grad
            # for resnets, self.norms are trainable LayerNorm
            self.norms = nn.ModuleList()
            for channel, scale in zip(channels, scales):
                self.norms.append(
                    nn.LayerNorm(
                        [channel, int(input_size[0] / scale),
                         int(input_size[1] / scale)],
                        elementwise_affine=True,
                    )
                )
        else:
            raise ValueError(
                f"feature_extractor_name {feature_extractor_name} is not supported. List of available feature_extractor_names are "
                "[cait_m48_448, deit_base_distilled_patch16_384, resnet18, wide_resnet50_2]."
            )

        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        self.feature_extractor.eval()

        self.fast_flow_blocks = nn.ModuleList()
        for channel, scale in zip(channels, scales):
            self.fast_flow_blocks.append(
                create_fast_flow_block(
                    input_dimensions=[channel, int(
                        input_size[0] / scale), int(input_size[1] / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                )
            )
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)

    def forward(self, input_tensor):
        # normalize
        input_tensor = self.normalize(input_tensor)

        # Extract features using the appropriate feature extractor
        if isinstance(self.feature_extractor, VisionTransformer):
            features = self._get_vit_features(input_tensor)
        elif isinstance(self.feature_extractor, Cait):
            features = self._get_cait_features(input_tensor)
        else:
            features = self._get_cnn_features(input_tensor)

        # Pass features through the fast flow blocks
        hidden_variables, log_jacobians = zip(
            *(fast_flow_block(feature) for fast_flow_block, feature in zip(self.fast_flow_blocks, features)))

        if not self.training:
            return self.anomaly_map_generator(hidden_variables)

        return hidden_variables, log_jacobians

    def _get_cnn_features(self, input_tensor):
        features = self.feature_extractor(input_tensor)
        features = [self.norms[i](feature)
                    for i, feature in enumerate(features)]

        return features

    def _get_cait_features(self, input_tensor):
        feature = self.feature_extractor.patch_embed(input_tensor)
        feature = feature + self.feature_extractor.pos_embed
        feature = self.feature_extractor.pos_drop(feature)

        for i in range(41):
            feature = self.feature_extractor.blocks[i](feature)

        batch_size, _, num_channels = feature.shape
        feature = self.feature_extractor.norm(feature)
        feature = feature.permute(0, 2, 1)
        feature = feature.reshape(
            batch_size, num_channels, self.input_size[0] // 16, self.input_size[1] // 16)
        features = [feature]

        return features

    def _get_vit_features(self, input_tensor):
        feature = self.feature_extractor.patch_embed(input_tensor)
        cls_token = self.feature_extractor.cls_token.expand(
            feature.shape[0], -1, -1)

        if self.feature_extractor.dist_token is None:
            feature = torch.cat((cls_token, feature), dim=1)
        else:
            feature = torch.cat(
                (
                    cls_token,
                    self.feature_extractor.dist_token.expand(
                        feature.shape[0], -1, -1),
                    feature,
                ),
                dim=1,
            )
        feature = self.feature_extractor.pos_drop(
            feature + self.feature_extractor.pos_embed)

        for i in range(8):  # paper Table 6. Block Index = 7
            feature = self.feature_extractor.blocks[i](feature)

        feature = self.feature_extractor.norm(feature)
        feature = feature[:, 2:, :]
        batch_size, _, num_channels = feature.shape
        feature = feature.permute(0, 2, 1)
        feature = feature.reshape(
            batch_size, num_channels, self.input_size[0] // 16, self.input_size[1] // 16)
        features = [feature]

        return features
