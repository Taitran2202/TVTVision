import torch.nn as nn
from torch import Tensor
import torchvision.transforms as transforms
from .module import FeatureExtractor, AnomalyMapGenerator, MMR_Student
from utils.pre_processing import Tiler


class MMR(nn.Module):
    def __init__(self, layers: list[str], input_size: tuple[int, int], backbone: str, patch_size: int,
                 in_chans: int, embed_dim: int, depth: int, num_heads: int, mlp_ratio: float,
                 scale_factors: tuple[float, float], fpn_output_dim: tuple[int, int],
                 use_feature_pooler: bool = True, use_tiler: bool = False, tiler_size: int = 64, stride: int = 64
                 ) -> None:
        super(MMR, self).__init__()
        self.use_feature_pooler = use_feature_pooler
        self.use_tiler = use_tiler
        self.tiler = Tiler(tile_size=tiler_size, stride=stride)
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.backbone = backbone
        self.teacher_model = FeatureExtractor(
            backbone=self.backbone, pre_trained=True, layers=layers)
        self.student_model = MMR_Student(
            input_size=input_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            layers=layers,
            scale_factors=scale_factors,
            fpn_output_dim=fpn_output_dim
        )

        for parameters in self.teacher_model.parameters():
            parameters.requires_grad = False

        self.feature_pooler = nn.AvgPool2d(3, 1, 1)
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)

    def forward(self, images: Tensor):
        images = self.normalize(images)
        if self.use_tiler:
            images = self.tiler.tile(images)

        teacher_features: dict[str, Tensor] = self.teacher_model(images)
        student_features: dict[str, Tensor] = self.student_model(images)

        if self.use_tiler:
            for i, features in enumerate(teacher_features):
                teacher_features[i] = self.tiler.untile(features)
            for i, features in enumerate(student_features):
                student_features[i] = self.tiler.untile(features)

        if self.use_feature_pooler:
            for i, feature in enumerate(teacher_features):
                teacher_features[i] = self.feature_pooler(feature)
            for i, feature in enumerate(student_features):
                student_features[i] = self.feature_pooler(feature)

        if self.training:
            return teacher_features, student_features
        else:
            return self.anomaly_map_generator(teacher_features=teacher_features, student_features=student_features)[:, 0, ...]
