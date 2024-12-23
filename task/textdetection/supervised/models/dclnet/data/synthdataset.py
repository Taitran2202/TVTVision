import os
import cv2
import numpy as np
import torch
import scipy.io as scio
import re
import ipdb
import itertools
from .module import *


class DCLNetSynthTextDataSet(DCLNetBaseDataset):
    def __init__(
        self, datadir: str, target: str, is_train: bool,
        scale: float = 0.25, length: int = 512,
        min_side: int = 640, max_side: int = 1920
    ):
        super(DCLNetSynthTextDataSet, self).__init__(
            datadir, target, is_train, scale, length, min_side, max_side)

        self.file_list, self.char_bbox, self.img_words = self.load_data()

    def load_data(self):
        gt = scio.loadmat(os.path.join(self.datadir, self.target, "gt.mat"))

        if self.is_train:
            file_list = gt["imnames"][0][:-100]
            img_words = gt["txt"][0][:-100]
            img_bbox = gt["wordBB"][0][:-100]
        else:
            file_list = gt["imnames"][0][-100:]
            img_words = gt["txt"][0][-100:]
            img_bbox = gt["wordBB"][0][-100:]

        return file_list, img_bbox, img_words

    def _wordBB2vertices(self, wordBB):
        vertices = []
        for i in range(wordBB.shape[-1]):
            vertices.append(wordBB[:, :, i].transpose().reshape(-1))
        return vertices

    def make_gt_score(self, index):
        img_path = os.path.join(
            self.datadir, self.target, self.file_list[index][0])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        wordBB = self.char_bbox[index].reshape((2, 4, -1))  # 2x4xn
        vertices = self._wordBB2vertices(wordBB)  # [ndarray(8,), ...]
        is_valid = np.ones(len(vertices))
        image, vertices = resize_img(
            image, vertices, self.min_side, self.max_side)
        image, vertices, is_valid = crop_img(
            image, vertices, self.length, is_valid)

        label_text, label_ignore, label_rho, label_theta = generate_label(
            image, vertices, is_valid, self.scale)
        label_text, label_ignore, label_rho, label_theta = list(map(
            lambda x: torch.Tensor(x), [label_text, label_ignore, label_rho, label_theta]))

        return image, label_text, label_ignore, label_rho, label_theta

    def load_synthtext_gt(self, index):
        img_path = os.path.join(
            self.datadir, self.target, self.file_list[index][0])
        try:
            wordbox = self.char_bbox[index].transpose((2, 1, 0))
        except:
            wordbox = np.expand_dims(self.char_bbox[index], axis=0)
            wordbox = wordbox.transpose((0, 2, 1))

        words = [re.split(" \n|\n |\n| ", t.strip())
                 for t in self.img_words[index]]
        words = list(itertools.chain(*words))
        words = [t for t in words if len(t) > 0]

        if len(words) != len(wordbox):
            ipdb.set_trace()

        single_img_bboxes = []
        for j in range(len(words)):
            box_info_dict = {"points": None, "text": None, "ignore": None}
            box_info_dict["points"] = wordbox[j]
            box_info_dict["text"] = words[j]
            box_info_dict["ignore"] = False
            single_img_bboxes.append(box_info_dict)

        return img_path, single_img_bboxes
