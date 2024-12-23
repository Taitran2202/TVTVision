# coding:utf-8
import os
import cv2
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .augmentation_utils import Transformer
from .annotation_utils import AnnotationReaderBase


class DatasetBase(Dataset):
    """Dataset base class """

    def __init__(self, datadir: str, target: str, is_train: bool, class_names, image_resize: int = 640, keep_difficult: bool = False,
                 use_mosaic=False, use_mixup=False, transformer: Transformer = None, color_transformer: Transformer = None):
        super(DatasetBase, self).__init__()
        # data
        self.datadir = datadir
        self.target = target
        self.class_names = class_names
        self.resize = image_resize
        self.keep_difficult = keep_difficult
        self.use_mosaic = use_mosaic
        self.use_mixup = use_mixup

        # mode
        self.is_train = is_train
        self.annotation_reader = AnnotationReaderBase()

        self.transformer = transformer
        self.color_transformer = color_transformer

        image_extensions = ['.jpeg', '.png', '.jpg', '.bmp']
        self.file_list = [file for file in glob(os.path.join(
            self.datadir, self.target, 'train/*' if is_train else 'test/*')) if os.path.splitext(file)[1].lower() in image_extensions]
        self.annotation_paths = []

        # convert to tensor
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index: int):
        if self.is_train:
            if self.use_mosaic and np.random.rand() > 0.7:
                image, bbox, label = self.make_mosaic(index)

                # mixup
                if self.use_mixup and np.random.rand() > 0.7:
                    index_ = np.random.randint(0, len(self))
                    image_, bbox_, label_ = self.make_mosaic(index_)
                    r = np.random.beta(8, 8)
                    image = (image*r+image_*(1-r)).astype(np.uint8)
                    bbox = np.vstack((bbox, bbox_))
                    label = np.hstack((label, label_))

                # Image enhancement
                if self.color_transformer:
                    image, bbox, label = self.color_transformer.transform(
                        image, bbox, label)

            else:
                image, bbox, label = self.load_image_target(index)

                if self.transformer:
                    image, bbox, label = self.transformer.transform(
                        image, bbox, label)

            target = {
                "boxes": bbox,
                "labels": label
            }
            return self.transform(image), target
        else:
            img_path, bbox, label = self.load_image_target(index)
            target = {
                "boxes": bbox,
                "labels": label
            }
            return img_path, target

    def make_mosaic(self, index: int):
        """ Create mosaic enhanced images """
        # Randomly select three images
        indexes = list(range(len(self.file_list)))
        choices = random.sample(indexes[:index]+indexes[index+1:], 3)
        choices.append(index)

        # Read in the four pictures and their labels used to make the mosaic
        images, bboxes, labels = [], [], []
        for i in choices:
            image, bbox, label = self.load_image_target(i)
            images.append(image)
            bboxes.append(bbox)
            labels.append(label)

        # Create a mosaic image and select splicing points
        img_size = self.resize
        mean = np.array([123, 117, 104])
        mosaic_img = np.ones((img_size*2, img_size*2, 3))*mean
        xc = int(random.uniform(img_size//2, 3*img_size//2))
        yc = int(random.uniform(img_size//2, 3*img_size//2))

        # Stitch images
        for i, (image, bbox, label) in enumerate(zip(images, bboxes, labels)):
            # Preserve/not preserve proportion scaling image
            ih, iw, _ = image.shape

            s = np.random.choice(np.arange(50, 210, 10))/100.
            if np.random.randint(2):
                r = img_size / max(ih, iw)
                if r != 1:
                    interp = cv2.INTER_LINEAR if (r > 1) else cv2.INTER_AREA
                    image = cv2.resize(
                        image, (int(iw*r*s), int(ih*r*s)), interpolation=interp)
            else:
                image = cv2.resize(image, (int(img_size*s), int(img_size*s)))

            # Paste the image to the upper left corner, upper right corner, lower left corner and lower right corner of the splicing point
            h, w, _ = image.shape
            if i == 0:
                # The coordinates of the upper left corner and lower right corner of the pasted part in the mosaic image
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                # The coordinates of the upper left corner and lower right corner of the pasted part in the original image
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(
                    yc - h, 0), min(xc + w, img_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(
                    xc - w, 0), yc, xc, min(img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:
                x1a, y1a, x2a, y2a = xc, yc, min(
                    xc + w, img_size * 2), min(img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            mosaic_img[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]

            # translate the coordinates
            dx = x1a - x1b
            dy = y1a - y1b
            bbox[:, [0, 2]] = (w * bbox[:, [0, 2]]) + dx
            bbox[:, [1, 3]] = (h * bbox[:, [1, 3]]) + dy

        # Handle bounding boxes beyond the coordinate system of the mosaic image
        bbox = np.clip(np.vstack(bboxes), 0, 2*img_size)
        label = np.hstack(labels)

        # Remove bounding boxes that are too small
        bbox_w = bbox[:, 2] - bbox[:, 0]
        bbox_h = bbox[:, 3] - bbox[:, 1]
        # mask = np.logical_and(bbox_w > 3, bbox_h > 3)
        mask = np.logical_and(bbox_w > 5, bbox_h > 5)
        bbox, label = bbox[mask], label[mask]
        if len(bbox) == 0:
            bbox = np.zeros((1, 4))
            label = np.array([0])

        # Normalized bounding box
        bbox /= mosaic_img.shape[0]

        return mosaic_img, bbox, label

    def load_image_target(self, index):
        file_path = self.file_list[index]

        # load an image
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # load an annotation
        anno = np.array(self.annotation_reader.read(
            self.annotation_paths[index]))

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
