import numpy as np
from typing import Tuple
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class EASTv2BaseDataset(Dataset):
    def __init__(
        self, datadir: str, target: str, is_train: bool,
        do_not_care_label: Tuple[str, str] = ['###', ''], min_text_size: float = 10,
        background_ratio: float = 0.125, min_crop_side_ratio: float = 0.1,
        img_size: int = 512, random_scale: list[float] = [0.5, 1, 2.0, 3.0]
    ):
        super(EASTv2BaseDataset, self).__init__()
        self.datadir = datadir
        self.target = target
        self.is_train = is_train
        self.do_not_care_label = do_not_care_label
        self.img_size = img_size
        self.random_scale = random_scale
        self.background_ratio = background_ratio
        self.min_text_size = min_text_size
        self.min_crop_side_ratio = min_crop_side_ratio

        self.file_list = []

        # convert ndarray into tensor
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if self.is_train:
            outs = self.make_gt_score(index)
            if outs is None:
                return self.__getitem__(np.random.randint(self.__len__()))

            image, score_map, geo_map, training_mask = outs
            score_map = score_map[np.newaxis, ::4, ::4].astype(np.float32)
            geo_map = np.swapaxes(geo_map, 1, 2)
            geo_map = np.swapaxes(geo_map, 1, 0)
            geo_map = geo_map[:, ::4, ::4].astype(np.float32)
            training_mask = training_mask[np.newaxis, ::4, ::4]
            training_mask = training_mask.astype(np.float32)
            
            return self.image_transform(image),  torch.Tensor(score_map), torch.Tensor(geo_map), torch.Tensor(training_mask)
        else:
            img_path, single_img_bboxes = self.load_test_dataset_iou(index)

            return img_path, single_img_bboxes

    def load_test_dataset_iou(self, index):
        if self.target == 'synth':
            total_img_path, total_bboxes_gt = self.load_synthtext_gt(index)
        elif self.target == 'icdar17':
            total_img_path, total_bboxes_gt = self.load_icdar2017_gt(index)
        elif self.target == 'icdar15':
            total_img_path, total_bboxes_gt = self.load_icdar2015_gt(index)
        elif self.target == 'icdar13':
            total_img_path, total_bboxes_gt = self.load_icdar2013_gt(index)

        return total_img_path, total_bboxes_gt
