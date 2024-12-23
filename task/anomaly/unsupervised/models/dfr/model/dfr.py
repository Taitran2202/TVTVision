import timm
import random
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class FeatCAE(nn.Module):
    """Autoencoder."""

    def __init__(self, in_channels: int = 1000, latent_dim: int = 50, is_bn: bool = True):
        super(FeatCAE, self).__init__()

        # encoder
        layers = []
        layers += [nn.Conv2d(in_channels, (in_channels + 2 *
                             latent_dim) // 2, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=(in_channels +
                                      2 * latent_dim) // 2)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d((in_channels + 2 * latent_dim) //
                             2, 2 * latent_dim, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=2 * latent_dim)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(2 * latent_dim, latent_dim,
                             kernel_size=1, stride=1, padding=0)]

        self.encoder = nn.Sequential(*layers)

        # decoder
        layers = []
        layers += [nn.Conv2d(latent_dim, 2 * latent_dim,
                             kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=2 * latent_dim)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(2 * latent_dim, (in_channels + 2 *
                             latent_dim) // 2, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=(in_channels +
                                      2 * latent_dim) // 2)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d((in_channels + 2 * latent_dim) //
                             2, in_channels, kernel_size=1, stride=1, padding=0)]
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class DFR(nn.Module):
    def __init__(self, backbone: str,
                 resize: Tuple[int, int] = (224, 224)
                 ):
        super(DFR, self).__init__()
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.resize = resize
        self.feature_extractor = timm.create_model(
            backbone,
            pretrained=True,
            features_only=True
        )
        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False

        self.average_pooling = nn.AvgPool2d(kernel_size=(4, 4), stride=(4, 4))
        self.c0 = sum(self.feature_extractor.feature_info.channels()[:-1])
        self.cd = 40

    def forward(self, inputs):
        inputs = self.normalize(inputs)
        with torch.no_grad():
            features = self.feature_extractor(inputs)

        resized_features = [F.interpolate(
            feature, size=self.resize) for feature in features[:-1]]
        resized_features = torch.cat(resized_features, dim=1)
        resized_features = self.average_pooling(resized_features)
        autoencoder_output = self.autoencoder(resized_features)

        if self.training:
            loss = torch.mean((autoencoder_output - resized_features) ** 2)
            return loss
        else:
            anomaly_map = torch.mean(
                (autoencoder_output - resized_features) ** 2, dim=1)
            anomaly_map = F.interpolate(
                anomaly_map.unsqueeze(1),
                size=self.resize,
                mode='bilinear',
                align_corners=True)

            return anomaly_map[:, 0, ...]

    def initialize_cd(self, dataloder):
        device = next(self.feature_extractor.parameters()).device

        extraction_per_sample = 20

        extractions = []
        with torch.no_grad():
            for i, (x, _) in enumerate(tqdm(dataloder)):
                x = self.normalize(x)
                x = x.to(device, dtype=torch.float32)
                features = self.feature_extractor(x)
            resized_features = [F.interpolate(
                feature, size=self.resize, mode='bilinear', align_corners=True) for feature in features[:-1]]
            resized_features = torch.cat(resized_features, dim=1)
            resized_features = self.average_pooling(resized_features)

            for feature in resized_features:
                for _ in range(extraction_per_sample):
                    row, col = random.randrange(
                        feature.shape[1]), random.randrange(feature.shape[2])
                    extraction = feature[row, col]
                    extractions.append(extraction.cpu().detach().numpy())

        extractions = np.array(extractions)
        print(f"Extractions Shape: {extractions.shape}")
        pca = PCA(0.9, svd_solver="full")
        pca.fit(extractions)
        self.cd = torch.tensor(
            pca.n_components_, requires_grad=False).to(device)
        self.autoencoder = FeatCAE(self.c0, self.cd).to(device)
        print(f"Components with explainable variance 0.9 -> {self.cd}")
