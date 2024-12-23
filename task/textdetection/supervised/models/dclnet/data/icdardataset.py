import os
import cv2
import numpy as np
import torch
from .module import *


class DCLNetICDARDataset(DCLNetBaseDataset):
    def __init__(
        self, datadir: str, target: str, is_train: bool,
        scale: float = 0.25, length: int = 512,
        min_side: int = 640, max_side: int = 1920
    ):
        super(DCLNetICDARDataset, self).__init__(
            datadir, target, is_train, scale, length, min_side, max_side)
        self.img_dir = os.path.join(
            self.datadir, self.target, 'train/images'if is_train else 'test/images')
        self.img_gt_box_dir = os.path.join(
            self.datadir, self.target, 'train/gt' if is_train else 'test/gt')
        self.file_list = os.listdir(self.img_dir)

    def make_gt_score(self, index):
        image, vertices, is_valid = self.load_data(index)
        image, vertices = resize_img(
            image, vertices, self.min_side, self.max_side)
        image, vertices, is_valid = crop_img(
            image, vertices, self.length, is_valid)

        label_text, label_ignore, label_rho, label_theta = generate_label(
            image, vertices, is_valid, self.scale)
        label_text, label_ignore, label_rho, label_theta = list(map(
            lambda x: torch.Tensor(x), [label_text, label_ignore, label_rho, label_theta]))

        return image, label_text, label_ignore, label_rho, label_theta

    def load_data(self, index):
        # read img
        img_name = self.file_list[index]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # read gt
        img_gt_box_path = os.path.join(
            self.img_gt_box_dir, "gt_%s.txt" % os.path.splitext(img_name)[0]
        )

        vertices, labels = self.load_img_gt_box(
            img_gt_box_path
        )

        return image, vertices, labels

    def load_img_gt_box(self, img_gt_box_path):
        lines = open(img_gt_box_path, encoding="utf-8").readlines()
        labels = []
        vertices = []
        for line in lines:
            vertices.append(adjust_box_sort(
                list(map(int, line.rstrip('\n').lstrip('\ufeff').split(',')[:8]))))
            label = 0 if '###' in line else 1
            labels.append(label)
        return np.array(vertices), np.array(labels)

    def load_icdar2015_gt(self, index):
        img_name = self.file_list[index]
        img_path = os.path.join(self.img_dir, img_name)

        gt_path = os.path.join(
            self.img_gt_box_dir, "gt_%s.txt" % os.path.splitext(img_name)[0]
        )

        lines = open(gt_path, encoding="utf-8").readlines()
        single_img_bboxes = []
        for line in lines:
            box_info_dict = {"points": None, "text": None, "ignore": None}

            box_info = line.strip().encode("utf-8").decode("utf-8-sig").split(",")
            box_points = [int(box_info[j]) for j in range(8)]
            word = box_info[8:]
            word = ",".join(word)
            box_points = np.array(box_points, np.int32).reshape(4, 2)
            box_info_dict["points"] = box_points
            box_info_dict["text"] = word
            if word == "###":
                box_info_dict["ignore"] = True
            else:
                box_info_dict["ignore"] = False

            single_img_bboxes.append(box_info_dict)

        return img_path, single_img_bboxes

    def load_icdar2013_gt(self, index):
        img_name = self.file_list[index]
        img_path = os.path.join(self.img_dir, img_name)

        gt_path = os.path.join(
            self.img_gt_box_dir, "gt_%s.txt" % os.path.splitext(img_name)[0]
        )

        lines = open(gt_path, encoding="utf-8").readlines()
        single_img_bboxes = []

        for line in lines:
            box_info_dict = {"points": None, "text": None, "ignore": None}

            box_info = line.strip().encode("utf-8").decode("utf-8-sig").split(",")
            box_points = [int(box_info[j]) for j in range(4)]
            word = box_info[4:]
            word = ",".join(word)
            box = [
                [box_points[0], box_points[1]],
                [box_points[2], box_points[1]],
                [box_points[2], box_points[3]],
                [box_points[0], box_points[3]],
            ]

            box_info_dict["points"] = box
            box_info_dict["text"] = word

            if word == "###":
                box_info_dict["ignore"] = True
            else:
                box_info_dict["ignore"] = False

            single_img_bboxes.append(box_info_dict)

        return img_path, single_img_bboxes
