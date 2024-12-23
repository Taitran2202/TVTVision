import cv2
import numpy as np
import torch
from .pse import pse
from utils.pre_processing.imgproc import normalization, resize_aspect_ratio


def test_net(model, image, device: str, kernel_num: int, min_area: float, min_score: float, bbox_type: str = 'rect', resize: int = 512):
    # resize
    img_resized, target_ratio, _ = resize_aspect_ratio(
        image, resize
    )
    ratio_h, ratio_w = 1 / target_ratio[0], 1 / target_ratio[1]

    # preprocessing
    img_resized = normalization(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        det_out, score_out = model(img_resized)

    score = score_out
    kernels = det_out[:, :kernel_num, :, :] > 0
    text_mask = kernels[:, :1, :, :]
    kernels[:, 1:, :, :] = kernels[:, 1:, :, :] * text_mask

    score = score.cpu().numpy()[0].astype(np.float32)
    kernels = kernels.cpu().numpy()[0].astype(np.uint8)

    # pse
    label = pse(kernels, min_area)

    label_num = np.max(label) + 1
    label = cv2.resize(
        label, (resize[1], resize[0]), interpolation=cv2.INTER_NEAREST)
    score = cv2.resize(
        score, (resize[1], resize[0]), interpolation=cv2.INTER_NEAREST)

    # Post-processing
    bboxes = []
    scores = []
    for i in range(1, label_num):
        ind = label == i
        points = np.array(np.where(ind)).transpose((1, 0))

        if points.shape[0] < min_area:
            label[ind] = 0
            continue

        score_i = np.mean(score[ind])
        if score_i < min_score:
            label[ind] = 0
            continue

        if bbox_type == 'rect':
            rect = cv2.minAreaRect(points[:, ::-1])
            bbox = cv2.boxPoints(rect) * (ratio_w, ratio_h)
        elif bbox_type == 'poly':
            binary = np.zeros(label.shape, dtype='uint8')
            binary[ind] = 1
            # bug in official released code
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bbox = contours[0] * (ratio_w, ratio_h)

        # điều chỉnh tọa độ
        bboxes.append(bbox)
        scores.append(score_i)

    return bboxes, scores


def test_onnx_net(model, image, kernel_num: int, min_area: float, min_score: float, bbox_type: str = 'rect', resize: int = 512):
    # resize
    img_resized, target_ratio, _ = resize_aspect_ratio(
        image, resize)
    ratio_h, ratio_w = 1 / target_ratio[0], 1 / target_ratio[1]

    # preprocessing
    img_resized = normalization(img_resized).numpy()
    img_resized = img_resized[np.newaxis, ...]

    # suy luận mô hình ONNX
    ort_inputs = {'input': img_resized}
    outputs = model.run(None, ort_inputs)
    ort_outputs, score_out = outputs
    score = score_out
    kernels = ort_outputs[:, :kernel_num, :, :] > 0
    text_mask = kernels[:, :1, :, :]
    kernels[:, 1:, :, :] = kernels[:, 1:, :, :] * text_mask

    score = score[0].astype(np.float32)
    kernels = kernels[0].astype(np.uint8)

    # pse
    label = pse(kernels, min_area)

    label_num = np.max(label) + 1
    label = cv2.resize(
        label, (resize[1], resize[0]), interpolation=cv2.INTER_NEAREST)
    score = cv2.resize(
        score, (resize[1], resize[0]), interpolation=cv2.INTER_NEAREST)

    # Post-processing
    bboxes = []
    scores = []
    for i in range(1, label_num):
        ind = label == i
        points = np.array(np.where(ind)).transpose((1, 0))

        if points.shape[0] < min_area:
            label[ind] = 0
            continue

        score_i = np.mean(score[ind])
        if score_i < min_score:
            label[ind] = 0
            continue

        if bbox_type == 'rect':
            rect = cv2.minAreaRect(points[:, ::-1])
            bbox = cv2.boxPoints(rect) * (ratio_w, ratio_h)
        elif bbox_type == 'poly':
            binary = np.zeros(label.shape, dtype='uint8')
            binary[ind] = 1
            # bug in official released code
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bbox = contours[0] * (ratio_w, ratio_h)

        # điều chỉnh tọa độ
        bboxes.append(bbox)
        scores.append(score_i)

    return bboxes, scores
