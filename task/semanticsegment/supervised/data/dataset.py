import os
import cv2
import albumentations as A
from glob import glob
from typing import Tuple
import torch
from torch.utils.data import Dataset, random_split
import torchvision.transforms as transforms


class SemanticSegmentDataset(Dataset):
    def __init__(
        self, datadir: str, target: str, is_train: bool,
        num_classes, color_map, num_train: float = 0.9,
        resize: Tuple[int, int] = (256, 256), transform=None
    ):
        super(SemanticSegmentDataset, self).__init__()
        # Mode
        self.num_classes = num_classes
        self.color_map = color_map
        self.is_train = is_train
        self.resize = resize
        self.transform = transform

        # load image file list
        self.datadir = datadir
        self.target = target

        image_extensions = ['.jpeg', '.png', '.jpg', '.bmp']
        self.file_list = [file for file in glob(os.path.join(
            self.datadir, self.target, 'images/*')) if os.path.splitext(file)[1].lower() in image_extensions]

        num_samples = len(self.file_list)
        num_train = int(num_samples * num_train)
        num_test = num_samples - num_train

        trainset, testset = random_split(self.file_list, [
                                         num_train, num_test], generator=torch.Generator().manual_seed(42))

        if self.is_train:
            self.file_list = trainset
        else:
            self.file_list = testset

        # Define augmentation pipeline
        train_transforms = [
            A.Resize(self.resize[0], self.resize[1])
        ]

        if self.transform['random_crop']['option']:
            train_transforms.append(
                A.RandomResizedCrop(
                    height=self.resize[0],
                    width=self.resize[1],
                    scale=self.transform['random_crop']['scale'],
                    ratio=self.transform['random_crop']['ratio'],
                    p=self.transform['random_crop']['prob']
                )
            )

        if self.transform['horizontal_flip']['option']:
            train_transforms.append(
                A.HorizontalFlip(p=self.transform['horizontal_flip']['prob'])
            )

        if self.transform['vertical_flip']['option']:
            train_transforms.append(
                A.VerticalFlip(p=self.transform['vertical_flip']['prob'])
            )

        if self.transform['color_distortion']['option']:
            train_transforms.append(
                A.ColorJitter(
                    brightness=self.transform['color_distortion']['brightness'],
                    contrast=self.transform['color_distortion']['contrast'],
                    saturation=self.transform['color_distortion']['saturation'],
                    hue=self.transform['color_distortion']['hue'],
                    p=self.transform['color_distortion']['prob']
                )
            )

        if self.transform['rotation']['option']:
            train_transforms.append(
                A.Rotate(
                    limit=self.transform['rotation']['degrees'],
                    border_mode=cv2.BORDER_REFLECT_101,
                    value=self.transform['rotation']['value_fill'],
                    p=self.transform['rotation']['prob']
                )
            )

        if self.transform['translation']['option']:
            train_transforms.append(
                A.ShiftScaleRotate(
                    shift_limit=self.transform['translation']['range'],
                    border_mode=cv2.BORDER_REFLECT_101,
                    value=self.transform['translation']['value_fill'],
                    p=self.transform['translation']['prob']
                )
            )

        # Define image transformation pipeline
        self.train_transform = A.Compose(train_transforms)

        self.val_transform = A.Compose(
            [
                A.Resize(self.resize[0], self.resize[1])
            ]
        )

        # convert to tensor
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]

        # image
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # mask
        mask = cv2.imread(
            file_path.replace('images', 'masks'),
            cv2.IMREAD_GRAYSCALE
        ).astype(bool).astype(int)

        transformed = self.train_transform(
            image=img, mask=mask) if self.is_train else self.val_transform(image=img, mask=mask)
        img = transformed["image"]
        mask = transformed["mask"]

        # convert ndarray into tensor
        img = self.transform(img)
        mask = torch.Tensor(mask).to(torch.int64)

        return img, mask