import os
import cv2
from typing import Tuple
import torch
from torch.utils.data import Dataset


class CRNNBaseDataset(Dataset):
    def __init__(
        self, datadir: str, target: str, train: bool, chars: str,
        resize: Tuple[int, int] = (512, 512),
    ):
        super(CRNNBaseDataset, self).__init__()
        self.chars = chars
        self.char2label = {char: i + 1 for i, char in enumerate(self.chars)}
        self.label2char = {label: char for char,
                           label in self.char2label.items()}

        # mode
        self.train = train
        self.resize = resize

        # load image file list
        self.datadir = datadir
        self.target = target

        self.file_list, self.texts = self.load_from_raw_files(
            os.path.join(self.datadir, self.target))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):

        img, target, target_length = self.make_img_text(index)

        return img, target, target_length

    def make_img_text(self, index):
        file_path = self.file_list[index]

        # image
        img = self.read_image(file_path)
        img = cv2.resize(img, dsize=(
            self.resize[1], self.resize[0]), interpolation=cv2.INTER_AREA)
        img = (img / 127.5) - 1.0

        # text
        text = self.texts[index]
        target = [self.char2label[c] for c in text]
        target_length = [len(target)]

        return torch.FloatTensor(img.reshape(1, img.shape[0], img.shape[1])), \
            torch.LongTensor(target), \
            torch.LongTensor(target_length)

    def read_image(self, file_path):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            alt_path = self.file_list[0]
            img = cv2.imread(alt_path, cv2.IMREAD_GRAYSCALE)

        return img
