import os
import cv2
import numpy as np
from glob import glob
import json
import math
import random
from typing import Tuple
import torch
from torch.utils.data import Dataset, random_split
import torchvision.transforms as transforms
from .utils import random_horizontal_flip, random_translate, random_colorjitter, gen_gaussian2d, detect_peaks


class HLSafeCountDataset(Dataset):
    def __init__(
        self, datadir: str, target: str, is_train: bool, aug, num_boxes: int = 6,
        num_train: float = 0.9, resize: Tuple[int, int] = (512, 512), out_stride: int = 4
    ):
        # mode
        self.is_train = is_train
        self.resize = resize
        self.aug = aug
        self.num_boxes = num_boxes
        self.out_stride = out_stride
        self.num_train = num_train

        # load image file list
        self.datadir = datadir
        self.target = target

        # load annotations
        self.annotations = self.load_annotations()
        num_samples = len(self.annotations)
        num_train = int(num_samples * num_train)
        num_test = num_samples - num_train
        trainset, testset = random_split(self.annotations, [
                                         num_train, num_test], generator=torch.Generator().manual_seed(42))

        if self.is_train:
            self.annotations = trainset
        else:
            self.annotations = testset

        self.file_list = [annotation[0] for annotation in self.annotations]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def load_annotations(self):
        image_extensions = ['.jpeg', '.png', '.jpg', '.bmp']
        imagesPath = [file for file in glob(os.path.join(
            self.datadir, self.target, 'images/*')) if os.path.splitext(file)[1].lower() in image_extensions]
        annotations = []
        for img_path in imagesPath:
            filename, _ = os.path.splitext(os.path.basename(img_path))
            annotation_path = os.path.join(
                self.datadir, self.target, 'annotations/' + filename + '.txt')
            bboxes = []
            if os.path.exists(annotation_path):
                with open(annotation_path) as json_file:
                    data = json.load(json_file)
                    bboxes = [value for p in data for value in [int(p['x']), int(p['y']), int(
                        p['x'])+int(p['w']), int(p['y'])+int(p['h'])] if int(p['w']) > 0 and int(p['h']) > 0]
            if len(bboxes) > 0:
                annotations.append([img_path, *bboxes])
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_path, bboxes = self.parse_annotation(idx)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        bboxes = bboxes.astype(np.float32)

        if self.is_train:
            if self.aug['random_horizontal_flip']['option']:
                image, bboxes = random_horizontal_flip(image, np.copy(bboxes))

            if self.aug['random_translate']['option']:
                image, bboxes = random_translate(image, np.copy(bboxes))

            if self.aug['random_colorjitter']['option']:
                image = random_colorjitter(image,
                                           brightness=self.aug['random_colorjitter']['brightness'],
                                           contrast=self.aug['random_colorjitter']['contrast'],
                                           saturation=self.aug['random_colorjitter']['saturation'],
                                           hue=self.aug['random_colorjitter']['hue'],
                                           p=self.aug['random_colorjitter']['prob']
                                           )

            image = cv2.resize(image, dsize=(self.resize[1], self.resize[0]))
            h1, w1, _ = image.shape

            scale_h = h1 / h
            scale_w = w1 / w

            bboxes[:, [0, 2]] *= scale_w
            bboxes[:, [1, 3]] *= scale_h

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
            bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]

            bboxes = bboxes / self.out_stride

            GAM = np.zeros((1, h1 // self.out_stride, w1 //
                           self.out_stride), dtype=np.float32)

            for bbox in bboxes:
                bbox = np.asarray(bbox, dtype=np.int32)
                dhsizew = int(bbox[2] / 2)
                dhsizeh = int(bbox[3] / 2)

                sigma = np.sqrt(dhsizew * dhsizeh) / (1.96*1.5)
                h_gauss = np.array(gen_gaussian2d(
                    dhsizew, dhsizeh, sigma, math.ceil(dhsizew / 4), math.ceil(dhsizeh / 4)))
                h_gauss = h_gauss / np.max(h_gauss)

                cmin = bbox[1]
                rmin = bbox[0]
                cmax = bbox[1] + int(2*dhsizeh)+1
                rmax = bbox[0] + int(2*dhsizew)+1

                if cmax > int(h1/self.out_stride):
                    cmax = int(h1/self.out_stride)

                if rmax > int(w1/self.out_stride):
                    rmax = int(w1/self.out_stride)

                GAM[0, cmin:cmax, rmin:rmax] = GAM[0, cmin:cmax,
                                                   rmin:rmax] + h_gauss[0:cmax-cmin, 0:rmax-rmin]

        else:
            image = cv2.resize(image, dsize=(self.resize[1], self.resize[0]))
            h1, w1, _ = image.shape

            scale_h = h1 / h
            scale_w = w1 / w

            bboxes[:, [0, 2]] *= scale_w
            bboxes[:, [1, 3]] *= scale_h
            bboxes = bboxes / self.out_stride

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
            bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]

            GAM = np.zeros((1, h1 // self.out_stride, w1 //
                           self.out_stride), dtype=np.float32)

            for bbox in bboxes:
                bbox = np.asarray(bbox, dtype=np.int32)

                dhsizew = int(bbox[2] / 2)
                dhsizeh = int(bbox[3] / 2)
                sigma = np.sqrt(dhsizew * dhsizeh) / (1.96*1.5)

                h_gauss = np.array(gen_gaussian2d(
                    dhsizew, dhsizeh, sigma, math.ceil(dhsizew / 4), math.ceil(dhsizeh / 4)))

                h_gauss = h_gauss / np.max(h_gauss)

                cmin = bbox[1]
                rmin = bbox[0]
                cmax = bbox[1] + int(2*dhsizeh)+1
                rmax = bbox[0] + int(2*dhsizew)+1

                if cmax > int(h1/self.out_stride):
                    cmax = int(h1/self.out_stride)

                if rmax > int(w1/self.out_stride):
                    rmax = int(w1/self.out_stride)

                GAM[0, cmin:cmax, rmin:rmax] = GAM[0, cmin:cmax,
                                                   rmin:rmax] + h_gauss[0:cmax-cmin, 0:rmax-rmin]

        GAM = np.clip(GAM, 0.0, 1.0)
        GAM = torch.tensor(GAM, dtype=torch.float32)
        bboxes[:, [2, 3]] = bboxes[:, [0, 1]] + bboxes[:, [2, 3]]
        bboxes[:, [0, 2]], bboxes[:, [1, 3]
                                  ] = bboxes[:, [1, 3]], bboxes[:, [0, 2]]
        chosen_boxes_idx = random.sample(
            range(bboxes.shape[0]), self.num_boxes)
        boxes = np.take(bboxes, chosen_boxes_idx, axis=0)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        n_obj = detect_peaks(GAM[0].numpy())

        image = self.transform(image)
        density_gt = torch.tensor(GAM, dtype=torch.float32)
        boxes = torch.tensor(boxes * self.out_stride, dtype=torch.float32)
        obj_count = torch.tensor(n_obj, dtype=torch.int64)
        return image, density_gt, boxes, obj_count

    def parse_annotation(self, idx):
        image_path, bboxes = self.annotations[idx][0], self.annotations[idx][1:]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        bboxes = np.array(bboxes).reshape(-1, 4)
        bboxes = bboxes[(bboxes[:, 2] - bboxes[:, 0] > 0) &
                        (bboxes[:, 3] - bboxes[:, 1] > 0)]
        if len(bboxes) == 0:
            rand_idx = int(random.random() * len(self))
            image_path, bboxes = self.parse_annotation(rand_idx)
        return image_path, bboxes
