import os
import cv2
import albumentations as A
from typing import Tuple
import torch
from torch.utils.data import random_split, Dataset
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, datadir: str, target: str, is_train: bool, aug,
                 num_train: float = 0.9, resize: Tuple[int, int] = (256, 256)):
        super(CustomDataset, self).__init__()
        self.datadir = datadir
        self.target = target
        self.is_train = is_train
        self.aug = aug
        self.resize = resize
        self.classes, self.class_to_idx = self._find_classes(
            os.path.join(self.datadir, self.target))
        self.file_list = self._make_dataset(os.path.join(
            self.datadir, self.target), self.class_to_idx)

        num_samples = len(self.file_list)
        num_train = int(num_samples * num_train)
        num_test = num_samples - num_train

        trainset, testset = random_split(self.file_list, [
                                         num_train, num_test], generator=torch.Generator().manual_seed(42))

        self.file_list = trainset if self.is_train else testset
        self.num_per_cls = self.calculate_num_per_cls(self.file_list)

        # Define augmentation pipeline
        train_transforms = [
            A.Resize(self.resize[0], self.resize[1])
        ]

        if self.aug['random_crop']['option']:
            train_transforms.append(
                A.RandomResizedCrop(
                    height=self.resize[0],
                    width=self.resize[1],
                    scale=self.aug['random_crop']['scale'],
                    ratio=self.aug['random_crop']['ratio'],
                    p=self.aug['random_crop']['prob']
                )
            )

        if self.aug['horizontal_flip']['option']:
            train_transforms.append(
                A.HorizontalFlip(p=self.aug['horizontal_flip']['prob'])
            )

        if self.aug['vertical_flip']['option']:
            train_transforms.append(
                A.VerticalFlip(p=self.aug['vertical_flip']['prob'])
            )

        if self.aug['color_distortion']['option']:
            train_transforms.append(
                A.ColorJitter(
                    brightness=self.aug['color_distortion']['brightness'],
                    contrast=self.aug['color_distortion']['contrast'],
                    saturation=self.aug['color_distortion']['saturation'],
                    hue=self.aug['color_distortion']['hue'],
                    p=self.aug['color_distortion']['prob']
                )
            )

        if self.aug['rotation']['option']:
            train_transforms.append(
                A.Rotate(
                    limit=self.aug['rotation']['degrees'],
                    border_mode=cv2.BORDER_REFLECT_101,
                    value=self.aug['rotation']['value_fill'],
                    p=self.aug['rotation']['prob']
                )
            )

        if self.aug['translation']['option']:
            train_transforms.append(
                A.ShiftScaleRotate(
                    shift_limit=self.aug['translation']['range'],
                    border_mode=cv2.BORDER_REFLECT_101,
                    value=self.aug['translation']['value_fill'],
                    p=self.aug['translation']['prob']
                )
            )

        # Define image transformation pipeline
        train_transforms.append(A.Resize(self.resize[0], self.resize[1]))
        self.train_transform = A.Compose(train_transforms)

        self.val_transform = A.Compose(
            [
                A.Resize(self.resize[0], self.resize[1])
            ]
        )

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        path, target = self.file_list[index]

        # Load the image from disk
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply any desired image transformations
        image = self.train_transform(image=image)[
            'image'] if self.is_train else self.val_transform(image=image)['image']

        return self.transform(image), target

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _make_dataset(self, dir, class_to_idx):
        images = []
        for target in sorted(class_to_idx.keys()):
            target_dir = os.path.join(dir, target)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

        return images

    def calculate_num_per_cls(self, dataset):
        num_per_cls = {cls: 0 for cls in self.classes}
        for _, target in dataset:
            class_name = self.classes[target]
            num_per_cls[class_name] += 1

        return num_per_cls
