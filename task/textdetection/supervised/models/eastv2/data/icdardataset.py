import os
import cv2
import numpy as np
from typing import Tuple
from .module import *


class EASTv2ICDARDataset(EASTv2BaseDataset):
    def __init__(
        self, datadir: str, target: str, is_train: bool,
        do_not_care_label: Tuple[str, str] = ['###', ''], min_text_size: float = 10,
        background_ratio: float = 0.125, min_crop_side_ratio: float = 0.1,
        img_size: int = 512, random_scale: list[float] = [0.5, 1, 2.0, 3.0]
    ):
        super(EASTv2ICDARDataset, self).__init__(datadir, target, is_train, do_not_care_label,
                                                 min_text_size, background_ratio, min_crop_side_ratio,  img_size, random_scale)
        self.img_dir = os.path.join(
            self.datadir, self.target, 'train/images'if is_train else 'test/images')
        self.img_gt_box_dir = os.path.join(
            self.datadir, self.target, 'train/gt' if is_train else 'test/gt')
        self.file_list = os.listdir(self.img_dir)

    def make_gt_score(self, index):
        image, text_polys, text_tags = self.load_data(index)
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

    def load_data(self, index):
        img_name = self.file_list[index]
        img_path = os.path.join(self.img_dir, img_name)

        # Kiểm tra và đọc ảnh
        image = cv2.imread(img_path)
        if image is None:
            # Nếu không đọc được ảnh, lấy ảnh từ vị trí index = 0
            index = 0
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
            if self.target == 'icdar13':
                box_info = line.strip().encode("utf-8").decode("utf-8-sig").split(" ")
            else:
                box_info = line.strip().encode("utf-8").decode("utf-8-sig").split(",")
            if len(box_info) > 8:
                box = np.asarray([int(box_info[i]) for i in range(8)])
                box = box.reshape(4,2)
            else:
                box_points = [int(box_info[j]) for j in range(4)]
                box =  np.asarray([box_points[0], box_points[1], box_points[2], box_points[1],
                       box_points[2], box_points[3], box_points[0], box_points[3]])
                box = box.reshape(4,2)
            vertices.append(adjust_box_sort(np.asarray(box)))
            label = 1 if '###' in line else 0
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
            box_info_dict["points"] = self.clockwise_sort(box_points)
            box_info_dict["text"] = word
            if word in self.do_not_care_label:
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

            box_info_dict["points"] = self.clockwise_sort(box)
            box_info_dict["text"] = word

            if word in self.do_not_care_label:
                box_info_dict["ignore"] = True
            else:
                box_info_dict["ignore"] = False

            single_img_bboxes.append(box_info_dict)

        return img_path, single_img_bboxes

    def load_icdar2017_gt(self, index):
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
            word = box_info[9:]
            word = ",".join(word)
            box_points = np.array(box_points, np.int32).reshape(4, 2)
            box_info_dict["points"] = self.clockwise_sort(box_points)
            box_info_dict["text"] = word
            if word in self.do_not_care_label:
                box_info_dict["ignore"] = True
            else:
                box_info_dict["ignore"] = False

            single_img_bboxes.append(box_info_dict)

        return img_path, single_img_bboxes

    def get_angle(self, point, base_point):
        # Tính góc tạo bởi hai điểm
        delta_x = point[0] - base_point[0]
        delta_y = point[1] - base_point[1]
        return np.arctan2(delta_y, delta_x)

    def clockwise_sort(self, points):
        # Tìm điểm cơ sở (base_point) bằng cách tìm tọa độ nhỏ nhất theo trục y
        base_point = min(points, key=lambda p: p[1])

        # Sắp xếp các điểm dựa trên góc của chúng so với điểm cơ sở
        sorted_points = sorted(
            points, key=lambda p: self.get_angle(p, base_point))

        return np.array(sorted_points)
