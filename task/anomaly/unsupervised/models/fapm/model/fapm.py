import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms as transforms
from .module import FeatureExtractor, AnomalyMapGenerator, KCenterGreedy
from utils.pre_processing import Tiler
from utils.filters import GaussianBlur2d


class FAPM(nn.Module):
    def __init__(
        self, input_size: tuple[int, int], layers: list[str], backbone: str = "wide_resnet50_2",
        pre_trained: bool = True, num_neighbors: int = 9, sampling_ratio: float = 0.1,
        tile_size: int = 64, stride: int = 64, sigma: int = 4, memory_bank_size=[]
    ):
        super(FAPM, self).__init__()
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.backbone = backbone
        self.layers = layers
        self.input_size = input_size
        self.num_neighbors = num_neighbors
        self.sampling_ratio = sampling_ratio
        self.feature_extractor = FeatureExtractor(
            backbone=self.backbone, pre_trained=pre_trained, layers=self.layers)
        self.feature_extractor.eval()
        self.feature_pooler = nn.AvgPool2d(3, 1, 1)
        self.register_buffer("memory_bank", Tensor(
            torch.zeros(memory_bank_size)))
        self.memory_bank: Tensor
        self.tiler = Tiler(tile_size=tile_size, stride=stride)
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=tile_size)
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(kernel_size=(
            kernel_size, kernel_size), sigma=(sigma, sigma), channels=1)

    def forward(self, input_tensor: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        input_tensor = self.normalize(input_tensor)
        input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            if self.backbone in ["resnet18", "wide_resnet50_2"]:
                with torch.no_grad():
                    features = self.feature_extractor(input_tensor)
                features = {layer: self.feature_pooler(
                    feature) for layer, feature in features.items()}
                embedding = self.generate_embedding(features)
            else:
                raise ValueError(
                    f"Backbone {self.backbone} is not supported. List of available backbones are "
                    "[resnet18, wide_resnet50_2]."
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
            anomaly_map = self.tiler.untile(anomaly_map)
            anomaly_map = self.blur(anomaly_map)
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
