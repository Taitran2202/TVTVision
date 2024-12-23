import os
import cv2
from glob import glob
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class TestDataset(Dataset):
    def __init__(
        self, cfg, datadir: str
    ):
        super(TestDataset, self).__init__()
        # mode
        self.mode = cfg['DATASET']['mode']

        # load image file list
        self.datadir = datadir
        image_extensions = ['.jpeg', '.png', '.jpg', '.bmp']
        self.file_list = [file for file in glob(os.path.join(
            self.datadir, '*')) if os.path.splitext(file)[1].lower() in image_extensions]
        self.resize = cfg['DATASET']['resize']
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]

        # image
        if self.mode:
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dsize=(self.resize[1], self.resize[0]))
            # convert ndarray into tensor
            img = self.transform(img)
            return img
        else:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, dsize=(
                self.resize[1], self.resize[0]), interpolation=cv2.INTER_AREA)
            img = (img / 127.5) - 1.0
            return torch.FloatTensor(img.reshape(1, img.shape[0], img.shape[1]))
