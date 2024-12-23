import numpy as np
import torch
from .yolo_utils import adjustResultCoordinates
from utils.pre_processing.imgproc import normalization, resize_aspect_ratio


def test_net(model, image, device: str, resize: int = 640):
    heigh, width = image.shape[:-1]

    # resize
    img_resized, target_ratio, _ = resize_aspect_ratio(
        image, resize
    )
    ratio_h, ratio_w = 1 / target_ratio[0], 1 / target_ratio[1]

    # preprocessing
    img_resized = normalization(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        bboxes, scores, labels = model(img_resized)

    # rescale
    bboxes = adjustResultCoordinates(bboxes, ratio_h, ratio_w, heigh, width)

    return bboxes, scores, labels


def test_onnx_net(model, image, resize: int = 640, postprocess=None):
    heigh, width = image.shape[:-1]

    # resize
    img_resized, target_ratio, _ = resize_aspect_ratio(
        image, resize)
    ratio_h, ratio_w = 1 / target_ratio[0], 1 / target_ratio[1]

    # preprocessing
    img_resized = normalization(img_resized).numpy()
    img_resized = img_resized[np.newaxis, ...]

    # suy luận mô hình ONNX
    ort_inputs = {'input': img_resized}
    ort_outputs = model.run(None, ort_inputs)

    # post process
    bboxes, scores, labels = postprocess(ort_outputs[0])

    # rescale
    boxes = adjustResultCoordinates(bboxes, ratio_h, ratio_w, heigh, width)

    return boxes, scores, labels
