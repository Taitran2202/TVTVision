import os
import cv2
import numpy as np
import re
import copy
import ipdb
import itertools
import scipy.io as scio
from typing import Tuple
from .utils import BaseDataset


class SynthTextDataSet(BaseDataset):
    def __init__(
        self, datadir: str, target: str, is_train: bool, aug,
        resize: Tuple[int, int] = (768, 768), gauss_init_size: int = 200,
        gauss_sigma: int = 40, enlarge_region: Tuple[float, float] = [0.5, 0.5],
        enlarge_affinity: Tuple[float, float] = [0.5, 0.5]
    ):
        super(SynthTextDataSet, self).__init__(datadir, target, is_train, aug, resize,
                                               gauss_init_size, gauss_sigma, enlarge_region, enlarge_affinity)

        self.file_list, self.char_bbox, self.img_words = self.load_data()

    def load_data(self):
        gt = scio.loadmat(os.path.join(self.datadir, self.target, "gt.mat"))

        if self.is_train:
            file_list = gt["imnames"][0][:-100]
            img_words = gt["txt"][0][:-100]
            img_bbox = gt["charBB"][0][:-100]
        else:
            file_list = gt["imnames"][0][-100:]
            img_words = gt["txt"][0][-100:]
            img_bbox = gt["wordBB"][0][-100:]

        return file_list, img_bbox, img_words

    def make_gt_score(self, index):
        img_path = os.path.join(
            self.datadir, self.target, self.file_list[index][0])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        all_char_bbox = copy.deepcopy(self.char_bbox[index]).transpose(
            (2, 1, 0)
        )  # shape : (Number of characters in image, 4, 2)

        img_h, img_w, _ = image.shape

        confidence_mask = np.ones((img_h, img_w), dtype=np.float32)

        words = [
            re.split(" \n|\n |\n| ", word.strip()) for word in self.img_words[index]
        ]
        words = list(itertools.chain(*words))
        words = [word for word in words if len(word) > 0]

        word_level_char_bbox = []
        char_idx = 0

        for i in range(len(words)):
            length_of_word = len(words[i])
            word_bbox = all_char_bbox[char_idx: char_idx + length_of_word]
            assert len(word_bbox) == length_of_word
            char_idx += length_of_word
            word_bbox = np.array(word_bbox)
            word_level_char_bbox.append(word_bbox)

        region_score = self.gaussian_builder.generate_region(
            img_h,
            img_w,
            word_level_char_bbox,
            horizontal_text_bools=[True for _ in range(len(words))],
        )

        affinity_score, all_affinity_bbox = self.gaussian_builder.generate_affinity(
            img_h,
            img_w,
            word_level_char_bbox,
            horizontal_text_bools=[True for _ in range(len(words))],
        )

        return image, region_score, affinity_score, confidence_mask, word_level_char_bbox, all_affinity_bbox, words

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
