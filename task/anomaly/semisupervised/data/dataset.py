import cv2
import os
import numpy as np
from glob import glob
from typing import Tuple
from torch.utils.data import Dataset
from ..transform import Transforms


class MVTecDataset(Dataset):
    def __init__(
        self, datadir: str, target: str, is_train: bool,
        resize: Tuple[int, int] = (256, 256), transform=None
    ):
        super(MVTecDataset, self).__init__()
        # Mode
        self.is_train = is_train
        self.transform = transform if transform else Transforms()

        # load image file list
        self.datadir = datadir
        self.target = target

        image_extensions = ['.jpeg', '.png', '.jpg', '.bmp']
        self.file_list = [file for file in glob(os.path.join(
            self.datadir, self.target, 'train/*/*' if is_train else 'test/*/*')) if os.path.splitext(file)[1].lower() in image_extensions]

        self.resize = resize

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        img, target = self.load_image_target(file_path)

        # augment
        img, target = self.transform(img, target)

        return img, target

    def load_image_target(self, file_path):
        # load an image
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(self.resize[1], self.resize[0]))

        # label
        label = 0 if 'good' in file_path else 1

        # mask
        if 'good' in file_path:
            mask = np.zeros(self.resize, dtype=np.float32)
        else:
            file_extension = os.path.splitext(file_path)[1].lower()
            # mask = cv2.imread(
            #     file_path.replace('test', 'ground_truth').replace(
            #         file_extension, f'_mask{file_extension}'),
            #     cv2.IMREAD_GRAYSCALE
            # )
            mask = cv2.imread(
                file_path.replace('test', 'ground_truth').replace(
                    file_extension, '_mask.png'),
                cv2.IMREAD_GRAYSCALE
            )
            mask = cv2.resize(mask, dsize=(
                self.resize[1], self.resize[0])).astype(bool).astype(int)

        target = {
            'mask': mask,
            'label': label
        }

        return img, target
