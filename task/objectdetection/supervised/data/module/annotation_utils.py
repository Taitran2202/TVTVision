# coding: utf-8
import numpy as np
from pathlib import Path
from typing import Dict, Union
from xml.etree import ElementTree as ET


class AnnotationReaderBase:
    """ Tag reader base class """

    def read(self, file_path: Union[str, Path]):
        """ Parse tag file

        Parameters
        ----------
        file_path: str
            file path

        Returns
        -------
        target: List[list] of shape `(n_objects, 5)`
            Label list, the first four elements of each label are the normalized bounding box, and the last label is the encoded category.
            e.g. `[[xmin, ymin, xmax, ymax, class], ...]`
        """
        raise NotImplementedError


class VocAnnotationReader(AnnotationReaderBase):
    """ XML format annotation converter """

    def __init__(self, class_names=None, keep_difficult=False):
        self.class_to_inddex = {k: v for v, k in enumerate(class_names)}
        self.keep_difficult = keep_difficult

    def read(self, file_path: str):
        root = ET.parse(file_path).getroot()

        # Image size
        img_size = root.find('size')
        w = int(img_size.find('width').text)
        h = int(img_size.find('height').text)

        # Extract all tags
        target = []
        for obj in root.iter('object'):
            # Is the sample difficult to predict?
            difficult = int(obj.find('difficult').text)
            if not self.keep_difficult and difficult:
                continue

            # Normalized box position
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            points = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(points):
                cur_pt = int(float(bbox.find(pt).text)) - 1
                # scale height or width
                cur_pt = cur_pt/w if i % 2 == 0 else cur_pt/h
                bndbox.append(cur_pt)

            # Check whether the data is legal
            if bndbox[0] >= bndbox[2] or bndbox[1] >= bndbox[3]:
                p = [int(bbox.find(pt).text) for pt in points]
                raise ValueError(
                    f"{file_path} Dirty data exists: object={name}, bbox={p}")

            label_idx = self.class_to_inddex[name]
            bndbox.append(label_idx)
            target.append(bndbox)  # [x1, y1, x2, y2, label_ind]

        return target  # [[x1, y1, x2, y2, label_ind], ... ]


class YoloAnnotationReader(AnnotationReaderBase):
    """ Yolo format dataset reader """

    def read(self, file_path: str):
        target = []

        # Read all bounding boxes
        with open(file_path) as f:
            lines = f.readlines()

        for line in lines:
            if not line.strip():
                continue
            c, cx, cy, w, h = line.strip().split()
            cx, cy, w, h = [float(i) for i in [cx, cy, w, h]]
            xmin = cx - w/2
            xmax = cx + w/2
            ymin = cy - h/2
            ymax = cy + h/2

            # Check data validity
            if xmin >= xmax or ymin >= ymax:
                raise ValueError(
                    f"{file_path} Dirty data exist: object={c}, bbox={[cx, cy, w, h]}")

            target.append([xmin, ymin, xmax, ymax, int(c)])

        return target


class SynthAnnotationReader(AnnotationReaderBase):
    """ Synth format dataset reader """

    def read(self, wordboxs: list[int]):

        # Tính toán xmin, ymin, xmax, ymax
        xmin = np.min(wordboxs[:, :, 0], axis=1)
        ymin = np.min(wordboxs[:, :, 1], axis=1)
        xmax = np.max(wordboxs[:, :, 0], axis=1)
        ymax = np.max(wordboxs[:, :, 1], axis=1)

        # Tạo NumPy array mới chứa xmin, ymin, xmax, ymax và cột số 0
        target = np.stack((xmin, ymin, xmax, ymax), axis=1)

        # Thêm cột số 0
        target = np.hstack((target, np.zeros((target.shape[0], 1))))

        return target
