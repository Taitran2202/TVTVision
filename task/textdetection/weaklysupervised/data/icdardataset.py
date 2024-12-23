import os
import cv2
import numpy as np
from typing import Tuple
from .utils import BaseDataset
from .utils.pseudo_label import PseudoCharBoxBuilder


class ICDARDataset(BaseDataset):
    def __init__(
        self, datadir: str, target: str, is_train: bool, aug, vis_train_dir: str,
        pseudo_vis_opt: bool, net,  do_not_care_label: Tuple[str, str] = ['###', ''], device: str = 'cpu',
        resize: Tuple[int, int] = (768, 768), gauss_init_size: int = 200,
        gauss_sigma: int = 40, enlarge_region: Tuple[float, float] = [0.5, 0.5],
        enlarge_affinity: Tuple[float, float] = [0.5, 0.5]
    ):
        super(ICDARDataset, self).__init__(datadir, target, is_train, aug,
                                           resize, gauss_init_size, gauss_sigma, enlarge_region, enlarge_affinity)
        self.pseudo_charbox_builder = PseudoCharBoxBuilder(
            vis_train_dir, pseudo_vis_opt, net, self.gaussian_builder, device)
        self.do_not_care_label = do_not_care_label
        self.img_dir = os.path.join(
            self.datadir, self.target, 'train/images'if is_train else 'test/images')
        self.img_gt_box_dir = os.path.join(
            self.datadir, self.target, 'train/gt' if is_train else 'test/gt')
        self.file_list = os.listdir(self.img_dir)

    def load_img_gt_box(self, img_gt_box_path):
        lines = open(img_gt_box_path, encoding="utf-8").readlines()
        word_bboxes = []
        words = []
        for line in lines:
            if self.target == 'icdar13':
                box_info = line.strip().encode("utf-8").decode("utf-8-sig").split(" ")
            else:
                box_info = line.strip().encode("utf-8").decode("utf-8-sig").split(",")
            if len(box_info) > 8 and self.target == 'icdar15':
                box_points = [int(box_info[i]) for i in range(8)]
                box_points = np.array(box_points, np.float32).reshape(4, 2)
                word = box_info[8:]
                word = ",".join(word)
            elif len(box_info) > 8 and self.target == 'icdar17':
                box_points = [int(box_info[i]) for i in range(8)]
                box_points = np.array(box_points, np.float32).reshape(4, 2)
                word = box_info[9:]
                word = ",".join(word)
            else:
                box_points = [int(box_info[j]) for j in range(4)]
                word = box_info[4:]
                word = ",".join(word).replace('"', "")
                box_points = [box_points[0], box_points[1], box_points[2], box_points[1],
                              box_points[2], box_points[3], box_points[0], box_points[3]]
                box_points = np.array(box_points, np.float32).reshape(4, 2)
            if word in self.do_not_care_label:
                words.append(self.do_not_care_label[0])
                word_bboxes.append(box_points)
                continue
            word_bboxes.append(box_points)
            words.append(word)

        return np.array(word_bboxes), words

    def load_data(self, index):
        img_name = self.file_list[index]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_gt_box_path = os.path.join(
            self.img_gt_box_dir, "gt_%s.txt" % os.path.splitext(img_name)[0]
        )
        word_bboxes, words = self.load_img_gt_box(
            img_gt_box_path
        )  # shape : (Number of word bbox, 4, 2)
        confidence_mask = np.ones((image.shape[0], image.shape[1]), np.float32)

        word_level_char_bbox = []
        do_care_words = []
        horizontal_text_bools = []

        if len(word_bboxes) == 0:
            return image, word_level_char_bbox, do_care_words, confidence_mask, horizontal_text_bools,

        _word_bboxes = word_bboxes.copy()
        for i in range(len(word_bboxes)):
            if words[i] in self.do_not_care_label:
                cv2.fillPoly(confidence_mask, [np.int32(_word_bboxes[i])], 0)
                continue

            pseudo_char_bbox, confidence, horizontal_text_bool = self.pseudo_charbox_builder.build_char_box(
                image, word_bboxes[i], words[i], img_name=img_name)

            cv2.fillPoly(confidence_mask, [
                         np.int32(_word_bboxes[i])], confidence)
            do_care_words.append(words[i])
            word_level_char_bbox.append(pseudo_char_bbox)
            horizontal_text_bools.append(horizontal_text_bool)

        return image, word_level_char_bbox, do_care_words, confidence_mask, horizontal_text_bools,

    def make_gt_score(self, index):
        """
        Make region, affinity scores using pseudo character-level GT bounding box
        word_level_char_bbox's shape : [word_num, [char_num_in_one_word, 4, 2]]
        :rtype region_score: np.float32
        :rtype affinity_score: np.float32
        :rtype confidence_mask: np.float32
        :rtype word_level_char_bbox: np.float32
        :rtype words: list
        """

        image, word_level_char_bbox, words, confidence_mask, horizontal_text_bools = self.load_data(
            index)
        img_h, img_w, _ = image.shape

        if len(word_level_char_bbox) == 0:
            region_score = np.zeros((img_h, img_w), dtype=np.float32)
            affinity_score = np.zeros((img_h, img_w), dtype=np.float32)
            all_affinity_bbox = []

        else:
            region_score = self.gaussian_builder.generate_region(
                img_h, img_w, word_level_char_bbox, horizontal_text_bools
            )
            affinity_score, all_affinity_bbox = self.gaussian_builder.generate_affinity(
                img_h, img_w, word_level_char_bbox, horizontal_text_bools
            )

        return image, region_score, affinity_score, confidence_mask, word_level_char_bbox, all_affinity_bbox, words

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
            box_info_dict["points"] = box_points
            box_info_dict["text"] = word
            if word == "###":
                box_info_dict["ignore"] = True
            else:
                box_info_dict["ignore"] = False

            single_img_bboxes.append(box_info_dict)

        return img_path, single_img_bboxes
