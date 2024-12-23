import torch.nn as nn
import torchvision.transforms as transforms
from .module import resnet18, resnet34, resnet50, wide_resnet50_2, HRNet, AnomalyMapGenerator
from utils.pre_processing import Tiler


class IKD(nn.Module):
    def __init__(self, backbone, input_size: tuple[int, int], use_feature_pooler: bool = True,
                 use_tiler: bool = False, tiler_size: int = 64, stride: int = 64):
        super(IKD, self).__init__()
        self.use_feature_pooler = use_feature_pooler
        self.use_tiler = use_tiler
        self.tiler = Tiler(tile_size=tiler_size, stride=stride)
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.input_size = input_size
        self.backbone = backbone
        if backbone in ['resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2']:
            # expert network
            self.model_expert = eval(f'{backbone}(pretrained=True)')
            # apprentice network
            self.model_apprentice = eval(f'{backbone}(pretrained=False)')
        elif backbone in ['hrnet18', 'hrnet32', 'hrnet48']:
            # expert network
            self.model_expert = HRNet(backbone, pretrained=True)
            # apprentice network
            self.model_apprentice = HRNet(backbone, pretrained=False)
        else:
            raise NotImplementedError

        for param in self.model_expert.parameters():
            param.requires_grad = False

        self.model_expert.eval()
        self.feature_pooler = nn.AvgPool2d(3, 1, 1)
        self.anomaly_map_generator = AnomalyMapGenerator(
            input_size=self.input_size)

    def forward(self, x):
        x = self.normalize(x)
        if self.use_tiler:
            x = self.tiler.tile(x)
        teacher_features = self.model_expert(x)
        student_features = self.model_apprentice(x)

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
