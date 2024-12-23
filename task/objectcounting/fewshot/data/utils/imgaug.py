import cv2
import numpy as np
import random


def random_translate(image, bboxes):
    if random.random() < 0.5:
        h, w, _ = image.shape
        max_bbox = np.concatenate(
            [np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

        M = np.array([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h))

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
    return image, bboxes


def random_horizontal_flip(image, bboxes):
    if random.random() < 0.5:
        _, w, _ = image.shape
        image = image[:, ::-1, :]
        bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
    return image, bboxes


def random_colorjitter(img, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.5):
    if np.random.uniform() < p:
        return img

    img = img.astype(np.float32) / 255.
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    brightness = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
    img[:, :, 2] = img[:, :, 2] * brightness

    contrast = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
    img[:, :, 2] = img[:, :, 2] * contrast + (1 - contrast)

    saturation = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
    img[:, :, 1] = img[:, :, 1] * saturation

    hue = np.random.uniform(-hue, hue)
    img[:, :, 0] = img[:, :, 0] + hue
    img[:, :, 0][img[:, :, 0] > 360] = img[:, :, 0][img[:, :, 0] > 360] - 360
    img[:, :, 0][img[:, :, 0] < 0] = img[:, :, 0][img[:, :, 0] < 0] + 360

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = img * 255
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)

    return img
