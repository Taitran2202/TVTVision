import timm
from tqdm import tqdm
from finch import FINCH
from sklearn.cluster import KMeans
from typing import List
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from .module import MultiScaleFusion, MultiSizeAttentionModule


class PRNet(nn.Module):
    def __init__(self, backbone: str, ratio: float = 0.1, num_classes: int = 2, input_size: tuple = (256, 256), layers: tuple = (1, 2, 3, 4)):
        super(PRNet, self).__init__()
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.backbone = backbone
        self.ratio = ratio
        self.encoder = timm.create_model(
            backbone,
            features_only=True,
            out_indices=layers,
            pretrained=True)

        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.encoder['layer4'].parameters():
            p.requires_grad = True

        self.encoder.eval()
        self.proto_features = nn.ParameterList()

        featuremap_dims = self.dryrun_find_featuremap_dims(
            self.encoder, input_size, len(layers))

        self.heights, self.widths, self.feature_dimensions = [], [], []
        for i in range(len(layers)):
            size = featuremap_dims[i]['resolution']
            self.heights.append(size[0])
            self.widths.append(size[1])
            self.feature_dimensions.append(featuremap_dims[i]['num_features'])

        self.ms_fuser1 = MultiScaleFusion(self.feature_dimensions[:-1])
        in_channels = [dim * 2 for dim in self.feature_dimensions[:-1]]
        self.ms_fuser2 = MultiScaleFusion(in_channels)
        self.attn_module = MultiSizeAttentionModule(
            in_channels, self.heights[:-1])

        self.up4_to_3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                      nn.Conv2d(
                                          self.feature_dimensions[3], self.feature_dimensions[2], kernel_size=3, padding=1),
                                      nn.BatchNorm2d(
                                          self.feature_dimensions[2]),
                                      nn.ReLU(inplace=True))
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(
                self.feature_dimensions[2] * 3, self.feature_dimensions[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feature_dimensions[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.feature_dimensions[2], self.feature_dimensions[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feature_dimensions[2]),
            nn.ReLU(inplace=True)
        )

        self.up3_to_2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                      nn.Conv2d(
                                          self.feature_dimensions[2], self.feature_dimensions[1], kernel_size=3, padding=1),
                                      nn.BatchNorm2d(
                                          self.feature_dimensions[1]),
                                      nn.ReLU(inplace=True))
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                self.feature_dimensions[1] * 3, self.feature_dimensions[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feature_dimensions[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.feature_dimensions[1], self.feature_dimensions[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feature_dimensions[1]),
            nn.ReLU(inplace=True)
        )

        self.up2_to_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                      nn.Conv2d(
                                          self.feature_dimensions[1], self.feature_dimensions[0], kernel_size=3, padding=1),
                                      nn.BatchNorm2d(
                                          self.feature_dimensions[0]),
                                      nn.ReLU(inplace=True))
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                self.feature_dimensions[0] * 3, self.feature_dimensions[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feature_dimensions[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.feature_dimensions[0], self.feature_dimensions[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feature_dimensions[0]),
            nn.ReLU(inplace=True)
        )

        self.up1_to_0 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                                      nn.Conv2d(
                                          self.feature_dimensions[0], self.feature_dimensions[0], kernel_size=3, padding=1),
                                      nn.BatchNorm2d(
                                          self.feature_dimensions[0]),
                                      nn.ReLU(inplace=True))

        self.conv_out = nn.Conv2d(
            self.feature_dimensions[0], num_classes, kernel_size=3, padding=1)

    def dryrun_find_featuremap_dims(
        self,
        feature_extractor,
        input_size: tuple[int, int],
        num_layers: list[str],
        device: torch.device = torch.device('cuda')
    ) -> dict[str, int | tuple[int, int]]:
        device = next(self.encoder.parameters()).device
        dryrun_input = torch.empty(1, 3, *input_size, device=device)
        dryrun_features = feature_extractor(dryrun_input)

        featuremap_dims = []
        for i in range(num_layers):
            featuremap_dims.append(
                {"num_features": dryrun_features[i].shape[1], "resolution": dryrun_features[i].shape[2:]})

        return featuremap_dims

    def initialize_proto_feature_maps(self, data_loader: DataLoader):
        device = next(self.encoder.parameters()).device

        layer1_features, layer2_features, layer3_features = [], [], []
        with torch.no_grad():
            for i, (x, _) in enumerate(tqdm(data_loader)):
                x = self.normalize(x)
                x = x.to(device, dtype=torch.float32)
                features = self.encoder(x)

                layer1_features.append(features[0].cpu())
                layer2_features.append(features[1].cpu())
                layer3_features.append(features[2].cpu())

        layer1_features = torch.cat(layer1_features, dim=0)
        layer2_features = torch.cat(layer2_features, dim=0)
        layer3_features = torch.cat(layer3_features, dim=0)
        _, C1, H1, W1 = layer1_features.shape
        _, C2, H2, W2 = layer2_features.shape
        _, C3, H3, W3 = layer3_features.shape
        layer1_features = layer1_features.view(-1, C1 * H1 * W1)
        layer2_features = layer2_features.view(-1, C2 * H2 * W2)
        layer3_features = layer3_features.view(-1, C3 * H3 * W3)

        # _, num_clust, _ = FINCH(
        #     layer1_features.cpu().numpy(), distance='euclidean')
        # K = sum(num_clust)
        K = int(len(data_loader.dataset) * self.ratio)

        kmeans = KMeans(n_clusters=K, random_state=0)
        kmeans.fit(layer1_features)
        layer1_features = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32)
        layer1_features = layer1_features.view(K, C1, H1, W1)

        kmeans = KMeans(n_clusters=K, random_state=0)
        kmeans.fit(layer2_features)
        layer2_features = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32)
        layer2_features = layer2_features.view(K, C2, H2, W2)

        kmeans = KMeans(n_clusters=K, random_state=0)
        kmeans.fit(layer3_features)
        layer3_features = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32)
        layer3_features = layer3_features.view(K, C3, H3, W3)

        self.proto_features = nn.ParameterList([
            nn.Parameter(layer1_features, requires_grad=False).to(device),
            nn.Parameter(layer2_features, requires_grad=False).to(device),
            nn.Parameter(layer3_features, requires_grad=False).to(device)
        ])

    def forward(self, images: Tensor):
        images = self.normalize(images)
        with torch.no_grad():
            features = self.encoder(images)

        layer4_features = features[-1]
        features = features[:-1]

        pfeatures = self.get_prototype_features(features, self.proto_features)
        rfeatures = self.get_residual_features(features, pfeatures)

        # multi-scale fusion
        features = self.ms_fuser1(*features)
        rfeatures = self.ms_fuser1(*rfeatures)

        # concatenate the input features and the residual features
        cfeatures = self.get_concatenated_features(features, rfeatures)

        # attention modules
        features = self.attn_module(cfeatures)
        features = self.ms_fuser2(*features)

        # decoder
        layer3_features = self.up4_to_3(layer4_features)
        layer3_features = torch.cat([features[2], layer3_features], dim=1)
        layer3_features = self.conv_block3(layer3_features)

        layer2_features = self.up3_to_2(layer3_features)
        layer2_features = torch.cat([features[1], layer2_features], dim=1)
        layer2_features = self.conv_block2(layer2_features)

        layer1_features = self.up2_to_1(layer2_features)
        layer1_features = torch.cat([features[0], layer1_features], dim=1)
        layer1_features = self.conv_block1(layer1_features)

        out_features = self.up1_to_0(layer1_features)
        predicted_mask = self.conv_out(out_features)

        if self.training:
            return torch.softmax(predicted_mask, dim=1)

        return 1 - torch.softmax(predicted_mask, dim=1)[:, 0, ...]

    def get_prototype_features(self, features: List[Tensor], proto_features: List[Tensor]) -> List[Tensor]:
        matched_proto_features = []
        for layer_id in range(len(features)):
            fi = features[layer_id]  # (B, dim, h, w)
            pi = proto_features[layer_id]  # (K, dim, h, w)
            B, C, H, W = fi.shape
            K, _, _, _ = pi.shape
            fir = fi.unsqueeze(1).expand(B, K, C, H, W).reshape(-1, C, H, W)
            pir = pi.unsqueeze(0).expand(B, K, C, H, W).reshape(-1, C, H, W)
            fir = fir.reshape(B * K, -1)
            pir = pir.reshape(B * K, -1)
            l2_dist = F.pairwise_distance(fir, pir, p=2)
            seps = l2_dist.chunk(B)
            cats = torch.stack(seps, dim=0)  # (B, K)
            inds = torch.argmin(cats, dim=1)  # (B, )
            matched_pi = pi[inds]  # (B, dim, h, w)
            matched_proto_features.append(matched_pi)

        return matched_proto_features

    def get_residual_features(self, features: List[Tensor], proto_features: List[Tensor]) -> List[Tensor]:
        residual_features = []
        for layer_id in range(len(features)):
            fi = features[layer_id]  # (B, dim, h, w)
            pi = proto_features[layer_id]  # (B, dim, h, w)

            ri = (fi - pi) ** 2
            # ri = F.mse_loss(fi, pi, reduction='none')
            residual_features.append(ri)

        return residual_features

    def get_concatenated_features(self, features1: List[Tensor], features2: List[Tensor]) -> List[Tensor]:
        cfeatures = []
        for layer_id in range(len(features1)):
            fi = features1[layer_id]  # (B, dim, h, w)
            pi = features2[layer_id]  # (B, dim, h, w)

            ci = torch.cat([fi, pi], dim=1)
            cfeatures.append(ci)

        return cfeatures
