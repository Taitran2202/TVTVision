import torch.nn as nn
from torch import Tensor
import torchvision.transforms as transforms
from .module import FeatureExtractor, AnomalyMapGenerator
from utils.pre_processing import Tiler


class STFPM(nn.Module):
    def __init__(self, layers: list[str], input_size: tuple[int, int], backbone: str = "resnet18",
                 use_feature_pooler: bool = True, use_tiler: bool = False, tiler_size: int = 64, stride: int = 64) -> None:
        super(STFPM, self).__init__()
        self.use_feature_pooler = use_feature_pooler
        self.use_tiler = use_tiler
        self.tiler = Tiler(tile_size=tiler_size, stride=stride)
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.backbone = backbone
        self.teacher_model = FeatureExtractor(
            backbone=self.backbone, pre_trained=True, layers=layers)
        self.student_model = FeatureExtractor(
            backbone=self.backbone, pre_trained=False, layers=layers, requires_grad=True
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
            for layer, data in teacher_features.items():
                teacher_features[layer] = self.tiler.untile(data)
            for layer, data in student_features.items():
                student_features[layer] = self.tiler.untile(data)

        if self.use_feature_pooler:
            for layer, data in teacher_features.items():
                teacher_features[layer] = self.feature_pooler(data)
            for layer, data in student_features.items():
                student_features[layer] = self.feature_pooler(data)

        if self.training:
            return teacher_features, student_features
        else:
            return self.anomaly_map_generator(teacher_features=teacher_features, student_features=student_features)[:, 0, ...]
