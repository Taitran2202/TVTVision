import os
from typing import Tuple
from .module import *


class PSEICDARDataset(PSEBaseDataset):
    def __init__(
        self, datadir: str, target: str, is_train: bool, aug,
        resize: Tuple[int, int] = (512, 512), short_size: int = 736,
        kernel_num: int = 7, min_scale: float = 0.7
    ):
        super(PSEICDARDataset, self).__init__(
            datadir, target, is_train, aug, resize, short_size, kernel_num, min_scale)

        self.img_dir = os.path.join(
            self.datadir, self.target, 'train/images'if is_train else 'test/images')
        self.img_gt_box_dir = os.path.join(
            self.datadir, self.target, 'train/gt' if is_train else 'test/gt')

        self.file_list = os.listdir(self.img_dir)

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

        bboxes, words = get_icdar_ann(image, img_gt_box_path, self.target)
        return image, bboxes, words

    def make_gt_score(self, index):
        image, bboxes, words = self.load_data(index)

        if bboxes.shape[0] > self.max_word_num:
            bboxes = bboxes[:self.max_word_num]
            words = words[:self.max_word_num]

        if self.aug['random_scale']['option']:
            image = random_scale(
                img=image,
                short_size=self.short_size,
                scales=self.aug['random_scale']['scales'],
                aspects=self.aug['random_scale']['aspects'],
                p=self.aug['random_scale']['prob']
            )

        gt_instance = np.zeros(image.shape[0:2], dtype='uint8')
        training_mask = np.ones(image.shape[0:2], dtype='uint8')
        if bboxes.shape[0] > 0:  # line
            bboxes = np.reshape(bboxes * ([image.shape[1], image.shape[0]] * 4),
                                (bboxes.shape[0], -1, 2)).astype('int32')
            for i in range(bboxes.shape[0]):
                cv2.drawContours(gt_instance, [bboxes[i]], -1, i + 1, -1)
                if words[i] == '###':
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)

        gt_kernels = []
        for i in range(1, self.kernel_num):
            rate = 1.0 - (1.0 - self.min_scale) / (self.kernel_num - 1) * i
            gt_kernel = np.zeros(image.shape[0:2], dtype='uint8')
            kernel_bboxes = shrink(bboxes, rate)
            for i in range(bboxes.shape[0]):
                cv2.drawContours(
                    gt_kernel, [kernel_bboxes[i].astype(int)], -1, 1, -1)
            gt_kernels.append(gt_kernel)

        imgs = [image, gt_instance, training_mask]
        imgs.extend(gt_kernels)

        if self.aug['random_horizontal_flip']['option']:
            imgs = random_horizontal_flip(
                imgs=imgs,
                p=self.aug['random_horizontal_flip']['prob']
            )
        if self.aug['random_rotate']['option']:
            imgs = random_rotate(
                imgs=imgs,
                random_angle=self.aug['random_rotate']['max_angle'],
                p=self.aug['random_rotate']['prob']
            )
        imgs = random_crop_padding(imgs, self.resize)
        image, gt_instance, training_mask, gt_kernels = imgs[0], imgs[1], imgs[2], imgs[3:]

        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1
        gt_kernels = np.array(gt_kernels)

        if self.aug['random_colorjitter']['option']:
            image = random_colorjitter(img=image,
                                       brightness=self.aug['random_colorjitter']['brightness'],
                                       contrast=self.aug['random_colorjitter']['contrast'],
                                       saturation=self.aug['random_colorjitter']['saturation'],
                                       hue=self.aug['random_colorjitter']['hue'],
                                       p=self.aug['random_colorjitter']['prob'])

        return image, gt_text, gt_kernels, training_mask

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