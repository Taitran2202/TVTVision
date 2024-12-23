import os
import cv2
import numpy as np
from glob import glob
from typing import Tuple
import albumentations as A
import torch
from torch.utils.data import Dataset, random_split
import torchvision.transforms as transforms


class EdgeDataset(Dataset):
    def __init__(
        self, datadir: str, target: str, is_train: bool, num_train: float = 0.9,
        threshold: float = 0.3, resize: Tuple[int, int] = (256, 256)
    ):
        super(EdgeDataset, self).__init__()
        # Mode
        self.is_train = is_train
        self.resize = resize
        self.threshold = threshold * 256.

        # load image file list
        self.datadir = datadir
        self.target = target

        image_extensions = ['.jpeg', '.png', '.jpg', '.bmp']
        self.file_list = [file for file in glob(os.path.join(
            self.datadir, self.target, 'imgs/*')) if os.path.splitext(file)[1].lower() in image_extensions]

        num_samples = len(self.file_list)
        num_train = int(num_samples * num_train)
        num_test = num_samples - num_train

        trainset, testset = random_split(self.file_list, [
                                         num_train, num_test], generator=torch.Generator().manual_seed(42))

        if self.is_train:
            self.file_list = trainset
        else:
            self.file_list = testset

        self.augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
            A.Transpose(p=0.5),
        ])

        # convert to tensor
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        img_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # image
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(self.resize[1], self.resize[0]))

        # mask
        mask = cv2.imread(
            file_path.replace('imgs', 'gt').replace('.jpg', '.png'),
            cv2.IMREAD_GRAYSCALE
        )
        mask = cv2.resize(mask, dsize=(
            self.resize[1], self.resize[0])).astype(float)

        if self.is_train:
            augmented = self.augmentations(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # convert ndarray into tensor
        img = self.transform(img)

        mask = mask[np.newaxis, :, :]
        mask[mask == 0.] = 0
        mask[np.logical_and(mask > 0., mask < self.threshold)] = 2
        mask[mask >= self.threshold] = 1
        mask = torch.Tensor(mask).to(torch.float32)

        return img, mask, img_name
