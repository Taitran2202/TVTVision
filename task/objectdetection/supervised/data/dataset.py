import os
import cv2
import numpy as np
import scipy.io as scio
from glob import glob
from .module import Transformer, VocAnnotationReader, YoloAnnotationReader, SynthAnnotationReader, DatasetBase


class VocDataset(DatasetBase):

    def __init__(self, datadir: str, target: str, is_train: bool, class_names, image_resize: int = 640,
                 keep_difficult: bool = False, use_mosaic=False, use_mixup=False, transformer: Transformer = None,
                 color_transformer: Transformer = None):
        super(VocDataset, self).__init__(datadir, target, is_train, class_names, image_resize,
                                         keep_difficult, use_mosaic, use_mixup, transformer, color_transformer)

        self.annotation_reader = VocAnnotationReader(
            class_names=class_names, keep_difficult=keep_difficult)
        self.annotation_paths = [file for file in glob(os.path.join(
            self.datadir, self.target, 'train/*.xml' if is_train else 'test/*.xml'))]


class YOLODataset(DatasetBase):

    def __init__(self, datadir: str, target: str, is_train: bool, class_names, image_resize: int = 640,
                 keep_difficult: bool = False, use_mosaic=False, use_mixup=False, transformer: Transformer = None,
                 color_transformer: Transformer = None):
        super(YOLODataset, self).__init__(datadir, target, is_train, class_names, image_resize,
                                          keep_difficult, use_mosaic, use_mixup, transformer, color_transformer)

        self.annotation_reader = YoloAnnotationReader()
        self.annotation_paths = [file for file in glob(os.path.join(
            self.datadir, self.target, 'train/*.txt' if is_train else 'test/*.txt'))]


class SynthDataset(DatasetBase):

    def __init__(self, datadir: str, target: str, is_train: bool, class_names, image_resize: int = 640,
                 keep_difficult: bool = False, use_mosaic=False, use_mixup=False, transformer: Transformer = None,
                 color_transformer: Transformer = None):
        super(SynthDataset, self).__init__(datadir, target, is_train, class_names, image_resize,
                                           keep_difficult, use_mosaic, use_mixup, transformer, color_transformer)

        self.annotation_reader = SynthAnnotationReader()
        self.file_list, self.img_bbox, self.img_words = self.load_data()

    def load_data(self):
        gt = scio.loadmat(os.path.join(self.datadir, self.target, "gt.mat"))

        if self.is_train:
            file_list = gt["imnames"][0][:-2000]
            img_words = gt["txt"][0][:-2000]
            img_bbox = gt["wordBB"][0][:-2000]
        else:
            file_list = gt["imnames"][0][-2000:]
            img_words = gt["txt"][0][-2000:]
            img_bbox = gt["wordBB"][0][-2000:]

        return file_list, img_bbox, img_words

    def load_image_target(self, index):
        file_path = self.file_list[index][0]
        image_path = os.path.join(
            self.datadir, self.target, self.file_list[index][0])

        # load an image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        try:
            wordbox = self.img_bbox[index].transpose((2, 1, 0))
        except:
            wordbox = np.expand_dims(self.img_bbox[index], axis=0)
            wordbox = wordbox.transpose((0, 2, 1))

        # load an annotation
        anno = np.array(self.annotation_reader.read(wordbox))
        anno[:, 0] /= w  # Chia cho chiều rộng
        anno[:, 1] /= h  # Chia cho chiều cao
        anno[:, 2] /= w  # Chia cho chiều rộng
        anno[:, 3] /= h  # Chia cho chiều cao

        # guard against no boxes via resizing
        anno = np.array(anno).reshape(-1, 5)
        boxes = anno[:, :4]
        labels = anno[:, 4].astype(int)

        if self.is_train:
            return (image, boxes, labels)
        else:
            boxes[:, [0, 2]] *= w
            boxes[:, [1, 3]] *= h

            return (file_path, boxes, labels)
