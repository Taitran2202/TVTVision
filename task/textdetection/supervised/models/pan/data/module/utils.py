import cv2
import numpy as np
import math
import random
import pyclipper
from shapely.geometry import Polygon
import string


def scale_aligned(img, h_scale, w_scale):
    h, w = img.shape[0:2]
    h = int(h * h_scale + 0.5)
    w = int(w * w_scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def random_scale(img, short_size=736, scales=[0.5, 2.0], aspects=[0.9, 1.1], p=0.5):
    if np.random.rand() > p:
        h, w = img.shape[0:2]

        scale = np.random.choice(np.array(scales))
        scale = (scale * short_size) / min(h, w)

        aspect = np.random.choice(np.array(aspects))
        h_scale = scale * math.sqrt(aspect)
        w_scale = scale / math.sqrt(aspect)

        img = scale_aligned(img, h_scale, w_scale)

    return img


def random_colorjitter(img, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5):
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


def random_crop_padding(imgs, target_size):
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs

    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        # make sure to crop the text region
        tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        br[br < 0] = 0
        br[0] = min(br[0], h - t_h)
        br[1] = min(br[1], w - t_w)

        i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
        j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
    else:
        i = random.randint(0, h - t_h) if h - t_h > 0 else 0
        j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            s3_length = int(imgs[idx].shape[-1])
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT,
                                       value=tuple(0 for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(
                img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT, value=(0,))
        n_imgs.append(img_p)
    return n_imgs


def random_horizontal_flip(imgs, p=0.5):
    if random.random() > p:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs, random_angle=10, p=0.5):
    if np.random.rand() > p:
        max_angle = random_angle
        angle = random.random() * 2 * max_angle - max_angle
        for i in range(len(imgs)):
            img = imgs[i]
            w, h = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
            img_rotation = cv2.warpAffine(
                img, rotation_matrix, (h, w), flags=cv2.INTER_NEAREST)
            imgs[i] = img_rotation
    return imgs


def distances(a, b):
    return np.linalg.norm((a - b), ord=2, axis=0)


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += distances(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        polygon = Polygon(bbox)
        area = polygon.area
        peri = perimeter(bbox)

        try:
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            offset = min(
                int(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

            shrinked_bbox = pco.Execute(-offset)
            if len(shrinked_bbox) == 0:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bbox = np.array(shrinked_bbox[0])
            if shrinked_bbox.shape[0] <= 2:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bboxes.append(shrinked_bbox)
        except Exception as e:
            print(type(shrinked_bbox), shrinked_bbox)
            print('area:', area, 'peri:', peri)
            shrinked_bboxes.append(bbox)

    return shrinked_bboxes


def update_word_mask(instance, instance_before_crop, word_mask):
    labels = np.unique(instance)

    for label in labels:
        if label == 0:
            continue
        ind = instance == label
        if np.sum(ind) == 0:
            word_mask[label] = 0
            continue
        ind_before_crop = instance_before_crop == label
        # print(np.sum(ind), np.sum(ind_before_crop))
        if float(np.sum(ind)) / np.sum(ind_before_crop) > 0.9:
            continue
        word_mask[label] = 0

    return word_mask


def get_vocabulary(voc_type, EOS='EOS', PADDING='PAD', UNKNOWN='UNK'):
    if voc_type == 'LOWERCASE':
        voc = list(string.digits + string.ascii_lowercase)
    elif voc_type == 'ALLCASES':
        voc = list(string.digits + string.ascii_letters)
    elif voc_type == 'ALLCASES_SYMBOLS':
        voc = list(string.printable[:-5])
    else:
        raise KeyError(
            'voc_type must be one of "LOWERCASE", "ALLCASES", "ALLCASES_SYMBOLS"')

    # update the voc with specifical chars
    voc.append(EOS)
    voc.append(PADDING)
    voc.append(UNKNOWN)

    char2id = dict(zip(voc, range(len(voc))))
    id2char = dict(zip(range(len(voc)), voc))

    return voc, char2id, id2char


def get_synth_ann(img, gts, texts, index):
    bboxes = np.array(gts[index])
    bboxes = np.reshape(bboxes, (bboxes.shape[0], bboxes.shape[1], -1))
    bboxes = bboxes.transpose(2, 1, 0)
    bboxes = np.reshape(
        bboxes, (bboxes.shape[0], -1)) / ([img.shape[1], img.shape[0]] * 4)

    words = []  
    for text in texts[index]:
        text = text.replace('\n', ' ').replace('\r', ' ')
        words.extend([w for w in text.split(' ') if len(w) > 0])

    return bboxes, words


def get_icdar_ann(img, gt_path, target):
    h, w = img.shape[0:2]
    lines = open(gt_path, encoding="utf-8").readlines()
    bboxes = []
    words = []
    for line in lines:
        if target == 'icdar13':
            box_info = line.strip().encode("utf-8").decode("utf-8-sig").split(" ")
        else:
            box_info = line.strip().encode("utf-8").decode("utf-8-sig").split(",")
        if len(box_info) > 8:
            bbox = [int(box_info[i]) for i in range(8)]
        else:
            box_points = [int(box_info[j]) for j in range(4)]
            bbox = [box_points[0], box_points[1], box_points[2], box_points[1],
                    box_points[2], box_points[3], box_points[0], box_points[3]]
        bbox = np.array(bbox) / ([w * 1.0, h * 1.0] * 4)
        bboxes.append(bbox)

        if target == 'icdar13':
            word = box_info[4:]
            words.append(word[0])
        elif target == 'icdar15':
            word = box_info[8:]
            words.append(word[0])
        elif target == 'icdar17':
            word = box_info[9:]
            words.append(word[0])

    return np.array(bboxes), words


def get_ctw_ann(img, gt_path):
    h, w = img.shape[0:2]
    lines = open(gt_path, encoding="utf-8").readlines()
    bboxes = []
    words = []
    for line in lines:
        line = line.replace('\xef\xbb\xbf', '')
        gt = line.split(',')

        x1 = np.int64(gt[0])
        y1 = np.int64(gt[1])

        bbox = [np.int64(gt[i]) for i in range(4, 32)]
        bbox = np.asarray(bbox) + ([x1 * 1.0, y1 * 1.0] * 14)
        bbox = np.asarray(bbox) / ([w * 1.0, h * 1.0] * 14)

        bboxes.append(bbox)
        words.append('???')
    return bboxes, words
