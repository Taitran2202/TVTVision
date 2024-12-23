from typing import Dict, Callable
import timm
import torch
import torch.nn as nn
from torch import Tensor
import torchvision.transforms as transforms
from .module import Decoder, MSFF, DiffModule, ReconstructiveSubNetwork
from utils.pre_processing import Tiler


class AESeg(nn.Module):
    def __init__(self, feature_extractor_name, num_classes: int = 2, sspcab: bool = False, sspcab_lambda: float = 0.1,
                 use_se: bool = True, use_feature_pooler: bool = True, use_tiler: bool = False,
                 tiler_size: int = 64, stride: int = 64):
        super(AESeg, self).__init__()
        self.use_feature_pooler = use_feature_pooler
        self.use_tiler = use_tiler
        self.tiler = Tiler(tile_size=tiler_size, stride=stride)
        self.feature_extractor = timm.create_model(
            feature_extractor_name,
            pretrained=True,
            features_only=True
        )
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        for p in self.feature_extractor['layer4'].parameters():
            p.requires_grad = True

        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.reconstructive_subnetwork = ReconstructiveSubNetwork(
            sspcab=sspcab)
        self.diff = DiffModule()
        self.msff = MSFF(use_se)
        self.decoder = Decoder(num_classes=num_classes)
        self.feature_pooler = nn.AvgPool2d(3, 1, 1)

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

    def forward(self, inputs):
        # normalize
        inputs = self.normalize(inputs)

        # reconstruction
        reconstruction = self.reconstructive_subnetwork(inputs)

        if self.use_tiler:
            reconstructiont = self.tiler.tile(reconstruction)

        # extract features normal
        features = self.feature_extractor(reconstructiont)

        if self.use_tiler:
            for layer, feature in enumerate(features):
                features[layer] = self.tiler.untile(feature)

        if self.use_feature_pooler:
            for i, feature in enumerate(features[1:-1]):
                features[i+1] = self.feature_pooler(feature)

        f_normal = features[1:-1]

        # extract features
        if self.use_tiler:
            inputs = self.tiler.tile(inputs)

        features = self.feature_extractor(inputs)

        if self.use_tiler:
            for layer, feature in enumerate(features):
                features[layer] = self.tiler.untile(feature)

        if self.use_feature_pooler:
            for i, feature in enumerate(features[1:-1]):
                features[i+1] = self.feature_pooler(feature)

        f_in = features[0]
        f_out = features[-1]
        f_ii = features[1:-1]

        # extract concatenated information(CI)
        concat_features = self.diff(features_normal=f_normal, features=f_ii)

        # Multi-scale Feature Fusion(MSFF) Module
        msff_outputs = self.msff(features=concat_features)

        # decoder
        predicted_mask = self.decoder(
            encoder_output=f_out,
            concat_features=[f_in] + msff_outputs
        )

        if self.training:
            return reconstruction, torch.softmax(predicted_mask, dim=1)

        return 1 - torch.softmax(predicted_mask, dim=1)[:, 0, ...]
