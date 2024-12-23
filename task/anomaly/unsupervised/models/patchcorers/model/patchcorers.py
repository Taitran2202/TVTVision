import timm
from typing import List
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms as transforms
from .module import FeatureExtractor, AnomalyMapGenerator, KCenterGreedy


class PatchcoreRS(nn.Module):
    def __init__(
        self, input_size: tuple[int, int], layers: list[str], backbone: str = "wide_resnet50_2",
        pre_trained: bool = True, num_neighbors: int = 9, sampling_ratio: float = 0.1, memory_bank_size=[]
    ):
        super(PatchcoreRS, self).__init__()
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.backbone = backbone
        self.layers = layers
        self.input_size = input_size
        self.num_neighbors = num_neighbors
        self.sampling_ratio = sampling_ratio

        if backbone in ["resnet18", "wide_resnet50_2"]:
            self.feature_extractor = FeatureExtractor(
                backbone=self.backbone, pre_trained=pre_trained, layers=self.layers)
        elif backbone in ["cait_m48_448", "deit_base_distilled_patch16_384"]:
            self.feature_extractor = timm.create_model(
                backbone, pretrained=pre_trained)

        self.feature_extractor.eval()
        self.feature_pooler = nn.AvgPool2d(3, 1, 1)
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)

        self.register_buffer("memory_bank", Tensor(
            torch.zeros(memory_bank_size)))
        self.memory_bank: Tensor

    def forward(self, input_tensor: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        input_tensor = self.normalize(input_tensor)
        with torch.no_grad():
            if self.backbone in ["deit_base_distilled_patch16_384"]:
                features = self._get_vit_features(input_tensor)
                embedding = features
            elif self.backbone in ["cait_m48_448"]:
                features = self._get_cait_features(input_tensor)
                embedding = features
            elif self.backbone in ["resnet18", "wide_resnet50_2"]:
                with torch.no_grad():
                    features = self.feature_extractor(input_tensor)
                features = {layer: self.feature_pooler(
                    feature) for layer, feature in features.items()}
                embedding = self.generate_embedding(features)
            else:
                raise ValueError(
                    f"Backbone {self.backbone} is not supported. List of available backbones are "
                    "[cait_m48_448, deit_base_distilled_patch16_384, resnet18, wide_resnet50_2]."
                )

        feature_map_shape = embedding.shape[-2:]
        embedding = self.reshape_embedding(embedding)

        if self.training:
            output = embedding
        else:
            patch_scores = self.nearest_neighbors(
                embedding=embedding, n_neighbors=self.num_neighbors)
            anomaly_map, anomaly_score = self.anomaly_map_generator(
                patch_scores=patch_scores, feature_map_shape=feature_map_shape
            )
            output = (anomaly_map[:, 0, ...], anomaly_score)

        return output

    def generate_embedding(self, features: dict[str, Tensor]) -> Tensor:
        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(
                layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        return embeddings

    @staticmethod
    def reshape_embedding(embedding: Tensor) -> Tensor:
        embedding_size = embedding.size(1)
        embedding = embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)
        return embedding

    def subsample_embedding(self, embedding: Tensor) -> None:
        # Coreset Subsampling
        sampler = KCenterGreedy(embedding=embedding,
                                sampling_ratio=self.sampling_ratio)
        coreset = sampler.sample_coreset()
        self.memory_bank = coreset

    @staticmethod
    def euclidean_dist(x: Tensor, y: Tensor) -> Tensor:
        """
        Calculates pair-wise distance between row vectors in x and those in y.
        Replaces torch cdist with p=2, as cdist is not properly exported to onnx and openvino format.
        Resulting matrix is indexed by x vectors in rows and y vectors in columns.
        Args:
            x: input tensor 1
            y: input tensor 2
        Returns:
            Matrix of distances between row vectors in x and y.
        """
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)  # |x|
        y_norm = y.pow(2).sum(dim=-1, keepdim=True)  # |y|
        # row distance can be rewritten as sqrt(|x| - 2 * x @ y.T + |y|.T)
        res = x_norm - 2 * \
            torch.matmul(x, y.transpose(-2, -1)) + y_norm.transpose(-2, -1)
        res = res.clamp_min_(0).sqrt_()
        return res

    def nearest_neighbors(self, embedding: Tensor, n_neighbors: int) -> tuple[Tensor, Tensor]:
        distances = self.euclidean_dist(embedding, self.memory_bank)
        patch_scores, _ = distances.topk(k=n_neighbors, largest=False, dim=1)
        return patch_scores

    def _get_cait_features(self, input_tensor: Tensor) -> List[Tensor]:
        """Get Class-Attention-Image-Transformers (CaiT) features.

        Args:
            input_tensor (Tensor): Input Tensor.

        Returns:
            List[Tensor]: List of features.
        """
        feature = self.feature_extractor.patch_embed(input_tensor)
        feature = feature + self.feature_extractor.pos_embed
        feature = self.feature_extractor.pos_drop(feature)
        for i in range(41):  # paper Table 6. Block Index = 40
            feature = self.feature_extractor.blocks[i](feature)
        batch_size, _, num_channels = feature.shape
        feature = self.feature_extractor.norm(feature)
        feature = feature.permute(0, 2, 1)
        feature = feature.reshape(
            batch_size, num_channels, self.input_size[0] // 16, self.input_size[1] // 16)
        return feature

    def _get_vit_features(self, input_tensor: Tensor) -> List[Tensor]:
        """Get Vision Transformers (ViT) features.

        Args:
            input_tensor (Tensor): Input Tensor.

        Returns:
            List[Tensor]: List of features.
        """
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
        return feature
