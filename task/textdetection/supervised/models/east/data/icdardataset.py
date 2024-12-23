import os
import cv2
import numpy as np
from .module import *


class EASTICDARDataset(EASTBaseDataset):
    def __init__(
        self, datadir: str, target: str, is_train: bool,
        scale: float = 0.25, length: int = 512
    ):
        super(EASTICDARDataset, self).__init__(
            datadir, target, is_train, scale, length)
        self.img_dir = os.path.join(
            self.datadir, self.target, 'train/images'if is_train else 'test/images')
        self.img_gt_box_dir = os.path.join(
            self.datadir, self.target, 'train/gt' if is_train else 'test/gt')
        self.file_list = os.listdir(self.img_dir)

    def make_gt_score(self, index):
        image, vertices, labels = self.load_data(index)
        image, vertices = adjust_height(image, vertices)
        image, vertices = rotate_img(image, vertices)
        image, vertices = crop_img(image, vertices, labels, self.length)

        score_map, geo_map, ignored_map = get_score_geo(
            image, vertices, labels, self.scale, self.length)

        return image, score_map, geo_map, ignored_map

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
                box = [int(box_info[i]) for i in range(8)]
            else:
                box_points = [int(box_info[j]) for j in range(4)]
                box = [box_points[0], box_points[1], box_points[2], box_points[1],
                       box_points[2], box_points[3], box_points[0], box_points[3]]
            vertices.append(adjust_box_sort(box))
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
            box_info_dict["points"] = self.clockwise_sort(box_points)
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

            box_info_dict["points"] = self.clockwise_sort(box)
            box_info_dict["text"] = word

            if word == "###":
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
            if word == "###":
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
