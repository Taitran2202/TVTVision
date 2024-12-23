import os
import cv2
import numpy as np
import scipy.io as scio
import re
import ipdb
import itertools
from typing import Tuple
from .module import *


class EASTv2SynthTextDataSet(EASTv2BaseDataset):
    def __init__(
        self, datadir: str, target: str, is_train: bool,
        do_not_care_label: Tuple[str, str] = ['###', ''], min_text_size: float = 10,
        background_ratio: float = 0.125, min_crop_side_ratio: float = 0.1,
        img_size: int = 512, random_scale: list[float] = [0.5, 1, 2.0, 3.0]
    ):
        super(EASTv2SynthTextDataSet, self).__init__(datadir, target, is_train, do_not_care_label,
                                                     min_text_size, background_ratio, min_crop_side_ratio,  img_size, random_scale)

        self.file_list, self.char_bbox, self.img_words = self.load_data()

    def load_data(self):
        gt = scio.loadmat(os.path.join(self.datadir, self.target, "gt.mat"))

        if self.is_train:
            # file_list = gt["imnames"][0][:-100]
            # img_words = gt["txt"][0][:-100]
            # img_bbox = gt["wordBB"][0][:-100]
            file_list = gt["imnames"][0][:1000]
            img_words = gt["txt"][0][:1000]
            img_bbox = gt["wordBB"][0][:1000]
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

        text_polys, text_tags = self.parse_mat(self.char_bbox[index])

        if np.random.randint(2):
            image, text_polys = rotate_im_poly(image, text_polys)
        h, w, _ = image.shape
        text_polys, text_tags = check_and_validate_polys(
            text_polys, text_tags, h, w)

        # random scale this image
        rd_scale = np.random.choice(self.random_scale)
        image = cv2.resize(image, dsize=None, fx=rd_scale, fy=rd_scale)
        text_polys *= rd_scale

        if np.random.rand() < self.background_ratio:
            outs = crop_background_infor(
                image, text_polys, text_tags, self.img_size, self.min_crop_side_ratio)
        else:
            outs = crop_foreground_infor(
                image, text_polys, text_tags, self.img_size, self.min_text_size, self.min_crop_side_ratio)
        
        return outs

    def parse_mat(self, mat):
        labels = []
        vertices = []
        if len(mat.shape) < 3:
            number = 1
        else:
            number = mat.shape[2]
        for cell in range(number):
            if number == 1:
                x1 = mat[0][0]
                x2 = mat[0][1]
                x3 = mat[0][2]
                x4 = mat[0][3]

                y1 = mat[1][0]
                y2 = mat[1][1]
                y3 = mat[1][2]
                y4 = mat[1][3]
            else:
                x1 = mat[0][0][cell]
                x2 = mat[0][1][cell]
                x3 = mat[0][2][cell]
                x4 = mat[0][3][cell]

                y1 = mat[1][0][cell]
                y2 = mat[1][1][cell]
                y3 = mat[1][2][cell]
                y4 = mat[1][3][cell]

            oriented_box = [[int(x1), int(y1)], [int(x2), int(y2)], [
                int(x3), int(y3)], [int(x4), int(y4)]]
            oriented_box = adjust_box_sort(np.asarray(oriented_box))
            oriented_box = np.asarray(oriented_box)
            vertices.append(oriented_box)
            labels.append(0)

        return np.array(vertices), np.array(labels)

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
