import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from .module import Decoder, MSFF, MemoryBlock


class MemSeg(nn.Module):
    def __init__(self, feature_extractor_name, memoryset, use_se: bool = True, device='cpu',
                 use_feature_pooler: bool = True, num_classes: int = 3, nb_memory_sample: int = 30):
        super(MemSeg, self).__init__()
        self.use_feature_pooler = use_feature_pooler
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.feature_extractor = timm.create_model(
            feature_extractor_name,
            pretrained=True,
            features_only=True
        )
        self.feature_extractor.to(device)
        self.memory_bank = MemoryBlock(
            self.feature_extractor, memoryset, nb_memory_sample, use_feature_pooler, device)
        self.msff = MSFF(use_se)
        self.decoder = Decoder(num_classes=num_classes)
        self.feature_pooler = nn.AvgPool2d(3, 1, 1)

        for p in self.feature_extractor['layer4'].parameters():
            p.requires_grad = True

    def forward(self, inputs):
        # normalize
        inputs = self.normalize(inputs)

        # extract features
        features = self.feature_extractor(inputs)

        if self.use_feature_pooler:
            for i, feature in enumerate(features[1:-1]):
                features[i+1] = self.feature_pooler(feature)

        f_in = features[0]
        f_out = features[-1]
        f_ii = features[1:-1]

        # extract concatenated information(CI)
        concat_features = self.memory_bank(features=f_ii)

        # Multi-scale Feature Fusion(MSFF) Module
        msff_outputs = self.msff(features=concat_features)

        # decoder
        predicted_mask = self.decoder(
            encoder_output=f_out,
            concat_features=[f_in] + msff_outputs
        )

        if self.training:
            return torch.softmax(predicted_mask, dim=1)

        return 1 - torch.softmax(predicted_mask, dim=1)[:, 0, ...]
