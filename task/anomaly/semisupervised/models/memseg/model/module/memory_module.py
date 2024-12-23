import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms


class MemoryBlock(nn.Module):
    def __init__(self, feature_extractor, normal_dataset, nb_memory_sample=30, use_feature_pooler=False, device='cpu'):
        super(MemoryBlock, self).__init__()
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.feature_extractor = feature_extractor
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False
        self.feature_extractor.eval()
        self.feature_extractor.to(device)
        self.device = device
        self.memory_information = nn.ParameterDict()
        self.normal_dataset = normal_dataset
        self.nb_memory_sample = nb_memory_sample
        self.feature_pooler = nn.AvgPool2d(3, 1, 1)
        self.use_feature_pooler = use_feature_pooler
        self._enter_in()

    def _enter_in(self):
        self.feature_extractor.eval()

        samples_idx = np.arange(len(self.normal_dataset))
        np.random.shuffle(samples_idx)

        with torch.no_grad():
            for i in range(self.nb_memory_sample):
                input_normal, _ = self.normal_dataset[samples_idx[i]]
                input_normal = self.normalize(
                    input_normal.unsqueeze(0)).to(self.device)
                features = self.feature_extractor(input_normal)

                if self.use_feature_pooler:
                    for i, feature in enumerate(features):
                        features[i] = self.feature_pooler(feature)

                for i, features_l in enumerate(features[1:-1]):
                    if f'level{i}' not in self.memory_information.keys():
                        self.memory_information[f'level{i}'] = nn.Parameter(
                            features_l, requires_grad=True)
                    else:
                        self.memory_information[f'level{i}'] = torch.cat(
                            [self.memory_information[f'level{i}'], nn.Parameter(features_l, requires_grad=True)], dim=0)

    def _calc_diff(self, features):
        diff_bank = torch.zeros(features[0].size(
            0), self.nb_memory_sample).to(self.device)

        for l, level in enumerate(self.memory_information.keys()):
            for b_idx, features_b in enumerate(features[l]):
                with torch.no_grad():

                    diff = ((torch.repeat_interleave(features_b.unsqueeze(
                        0), repeats=self.nb_memory_sample, dim=0) - self.memory_information[level]) ** 2).mean(dim=[1, 2, 3])

                    diff_bank[b_idx] += diff

        return diff_bank

    def forward(self, features):
        diff_bank = self._calc_diff(features)

        for l, level in enumerate(self.memory_information.keys()):
            selected_features = torch.index_select(
                self.memory_information[level], dim=0, index=diff_bank.argmin(dim=1))
            diff_features = ((selected_features - features[l]) ** 2)
            features[l] = torch.cat([features[l], diff_features], dim=1)

        return features
