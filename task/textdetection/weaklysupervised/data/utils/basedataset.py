import cv2
from typing import Tuple
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .gaussian import GaussianBuilder
from .imgaug import random_scale, random_rotate, random_horizontal_flip, img_resize, random_colorjitter, random_crop


class BaseDataset(Dataset):
    def __init__(
        self, datadir: str, target: str, is_train: bool, aug,
        resize: Tuple[int, int] = (512, 512),
        gauss_init_size: int = 200, gauss_sigma: int = 40,
        enlarge_region: Tuple[float, float] = [0.5, 0.5],  # x axis, y axis,
        enlarge_affinity: Tuple[float, float] = [0.5, 0.5]  # x axis, y axis
    ):
        super(BaseDataset, self).__init__()
        self.datadir = datadir
        self.target = target
        self.is_train = is_train
        self.aug = aug
        self.resize = resize

        self.gaussian_builder = GaussianBuilder(
            init_size=gauss_init_size,
            sigma=gauss_sigma,
            enlarge_region=enlarge_region,
            enlarge_affinity=enlarge_affinity
        )

        # convert ndarray into tensor
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def augment_image(self, image, region_score, affinity_score, confidence_mask):
        augment_targets = [image, region_score,
                           affinity_score, confidence_mask]

        if self.aug['random_scale']['option']:
            augment_targets = random_scale(
                augment_targets,
                scale_range=self.aug['random_scale']['range'],
                p=self.aug['random_scale']['prob']
            )

        augment_targets = random_crop(augment_targets, self.resize[0])

        if self.aug['random_rotate']['option']:
            augment_targets = random_rotate(
                augment_targets,
                max_angle=self.aug['random_rotate']['max_angle'],
                p=self.aug['random_rotate']['prob']
            )

        if self.aug['random_horizontal_flip']['option']:
            augment_targets = random_horizontal_flip(
                augment_targets,
                p=self.aug['random_horizontal_flip']['prob']
            )

        if self.aug['random_colorjitter']['option']:
            image, region_score, affinity_score, confidence_mask = augment_targets
            image = random_colorjitter(image, brightness=self.aug['random_colorjitter']['brightness'],
                                       contrast=self.aug['random_colorjitter']['contrast'],
                                       saturation=self.aug['random_colorjitter']['saturation'],
                                       hue=self.aug['random_colorjitter']['hue'],
                                       p=self.aug['random_colorjitter']['prob'])
        else:
            image, region_score, affinity_score, confidence_mask = augment_targets

        return image, region_score, affinity_score, confidence_mask

    def resize_to_half(self, ground_truth, interpolation):
        return cv2.resize(
            ground_truth,
            (self.resize[0] // 2, self.resize[1] // 2),
            interpolation=interpolation,
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if self.is_train:
            image, region_score, affinity_score, confidence_mask, _, _, _ = self.make_gt_score(
                index)
            image, region_score, affinity_score, confidence_mask = self.augment_image(
                image, region_score, affinity_score, confidence_mask
            )

            # convert ndarray into tensor
            image = self.image_transform(image)
            region_score = self.resize_to_half(
                region_score, interpolation=cv2.INTER_CUBIC)
            affinity_score = self.resize_to_half(
                affinity_score, interpolation=cv2.INTER_CUBIC)
            confidence_mask = self.resize_to_half(
                confidence_mask, interpolation=cv2.INTER_NEAREST)

            region_score = torch.Tensor(region_score)
            affinity_score = torch.Tensor(affinity_score)
            confidence_mask = torch.Tensor(confidence_mask)

            return image, region_score, affinity_score, confidence_mask
        else:
            img_path, single_img_bboxes = self.load_test_dataset_iou(index)

            return img_path, single_img_bboxes

    def load_test_dataset_iou(self, index):
        if self.target == 'synth':
            img_path, single_img_bboxes = self.load_synthtext_gt(index)
        elif self.target == 'icdar17':
            img_path, single_img_bboxes = self.load_icdar2017_gt(index)
        elif self.target == 'icdar15':
            img_path, single_img_bboxes = self.load_icdar2015_gt(index)
        elif self.target == 'icdar13':
            img_path, single_img_bboxes = self.load_icdar2013_gt(index)

        return img_path, single_img_bboxes
