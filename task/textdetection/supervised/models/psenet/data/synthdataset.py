import os
import ipdb
import scipy.io as scio
import re
import itertools
from typing import Tuple
from .module import *


class PSESynthTextDataSet(PSEBaseDataset):
    def __init__(
        self, datadir: str, target: str, is_train: bool, aug,
        resize: Tuple[int, int] = (512, 512), short_size: int = 736,
        kernel_num: int = 7, min_scale: float = 0.7
    ):
        super(PSESynthTextDataSet, self).__init__(datadir, target,
                                                  is_train, aug, resize, short_size, kernel_num, min_scale)

        self.file_list, self.word_bbox, self.img_words = self.load_data()

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

    def make_gt_score(self, index):
        img_path = os.path.join(
            self.datadir, self.target, self.file_list[index][0])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes, words = get_synth_ann(
            image, self.word_bbox, self.img_words, index)

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
        if bboxes.shape[0] > 0:
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
            for i in range(len(bboxes)):
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

        max_instance = np.max(gt_instance)
        gt_bboxes = np.zeros((self.max_word_num, 4), dtype=np.int32)
        for i in range(1, max_instance + 1):
            ind = gt_instance == i
            if np.sum(ind) == 0:
                continue
            points = np.array(np.where(ind)).transpose((1, 0))
            tl = np.min(points, axis=0)
            br = np.max(points, axis=0) + 1
            gt_bboxes[i] = (tl[0], tl[1], br[0], br[1])

        if self.aug['random_colorjitter']['option']:
            image = random_colorjitter(img=image,
                                       brightness=self.aug['random_colorjitter']['brightness'],
                                       contrast=self.aug['random_colorjitter']['contrast'],
                                       saturation=self.aug['random_colorjitter']['saturation'],
                                       hue=self.aug['random_colorjitter']['hue'],
                                       p=self.aug['random_colorjitter']['prob'])

        return image, gt_text, gt_kernels, training_mask

    def load_synthtext_gt(self, index):
        img_path = os.path.join(
            self.datadir, self.target, self.file_list[index][0])
        try:
            wordbox = self.word_bbox[index].transpose((2, 1, 0))
        except:
            wordbox = np.expand_dims(self.word_bbox[index], axis=0)
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
