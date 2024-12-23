import cv2
import numpy as np
import random


def random_scale(targets, scale_range, p=0.5):
    if np.random.rand() > p:
        scale_factor = np.random.choice(scale_range)
        h, w = targets[0].shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        image = cv2.resize(targets[0], (new_w, new_h))
        region_score = cv2.resize(targets[1], (new_w, new_h))
        affinity_score = cv2.resize(targets[2], (new_w, new_h))
        confidence_mask = cv2.resize(targets[3], (new_w, new_h))

        augment_targets = [image, region_score,
                           affinity_score, confidence_mask]
    else:
        augment_targets = targets

    return augment_targets


def random_rotate(targets, max_angle, p=0.5):
    if random.random() > p:
        angle = random.uniform(-max_angle, max_angle)
        M = cv2.getRotationMatrix2D(
            (targets[0].shape[1] // 2, targets[0].shape[0] // 2), angle, 1)
        image = cv2.warpAffine(
            targets[0], M, targets[0].shape[1::-1], flags=cv2.INTER_LINEAR)
        region_score = cv2.warpAffine(
            targets[1], M, targets[1].shape[1::-1], flags=cv2.INTER_LINEAR)
        affinity_score = cv2.warpAffine(
            targets[2], M, targets[2].shape[1::-1], flags=cv2.INTER_LINEAR)
        confidence_mask = cv2.warpAffine(
            targets[3], M, targets[3].shape[1::-1], flags=cv2.INTER_LINEAR)

        augment_targets = [image, region_score,
                           affinity_score, confidence_mask]
    else:
        augment_targets = targets

    return augment_targets


def random_horizontal_flip(targets, p=0.5):
    if random.random() > p:
        image = cv2.flip(targets[0], 1)
        region_score = cv2.flip(targets[1], 1)
        affinity_score = cv2.flip(targets[2], 1)
        confidence_mask = cv2.flip(targets[3], 1)
    else:
        image = targets[0]
        region_score = targets[1]
        affinity_score = targets[2]
        confidence_mask = targets[3]

    augment_targets = [image, region_score, affinity_score, confidence_mask]
    return augment_targets


def img_resize(targets, size):
    image, region_score, affinity_score, confidence_mask = targets
    ratio = size[0] / max(image.shape[:2])
    img = cv2.resize(
        image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))

    region = cv2.resize(region_score, (img.shape[1], img.shape[0]))
    affinity = cv2.resize(affinity_score, (img.shape[1], img.shape[0]))
    confidence = cv2.resize(confidence_mask, (img.shape[1], img.shape[0]))

    padding_height = size[0] - img.shape[0]
    padding_width = size[1] - img.shape[1]

    img = cv2.copyMakeBorder(img, 0, padding_height, 0,
                             padding_width, cv2.BORDER_CONSTANT, value=0)
    region = cv2.copyMakeBorder(
        region, 0, padding_height, 0, padding_width, cv2.BORDER_CONSTANT, value=0)
    affinity = cv2.copyMakeBorder(
        affinity, 0, padding_height, 0, padding_width, cv2.BORDER_CONSTANT, value=0)
    confidence = cv2.copyMakeBorder(
        confidence, 0, padding_height, 0, padding_width, cv2.BORDER_CONSTANT, value=0)

    return [img, region, affinity, confidence]


def random_crop(targets, size):
    image, region_score, affinity_score, confidence_mask = targets

    h, w = image.shape[:2]
    x = random.randint(0, max(0, w - size))
    y = random.randint(0, max(0, h - size))

    cropped_image = np.zeros((size, size, image.shape[2]), dtype=image.dtype)
    cropped_region_score = np.zeros((size, size), dtype=region_score.dtype)
    cropped_affinity_score = np.zeros((size, size), dtype=affinity_score.dtype)
    cropped_confidence_mask = np.zeros(
        (size, size), dtype=confidence_mask.dtype)

    h_offset = min(h, size)
    w_offset = min(w, size)

    cropped_image[:h_offset, :w_offset] = image[y:y+h_offset, x:x+w_offset]
    cropped_region_score[:h_offset,
                         :w_offset] = region_score[y:y+h_offset, x:x+w_offset]
    cropped_affinity_score[:h_offset,
                           :w_offset] = affinity_score[y:y+h_offset, x:x+w_offset]
    cropped_confidence_mask[:h_offset,
                            :w_offset] = confidence_mask[y:y+h_offset, x:x+w_offset]

    cropped_targets = [cropped_image, cropped_region_score,
                       cropped_affinity_score, cropped_confidence_mask]

    return cropped_targets


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
