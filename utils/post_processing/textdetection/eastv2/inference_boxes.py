import numpy as np
import torch
from .east_utils import get_boxes, adjustResultCoordinates
from utils.pre_processing.imgproc import normalization, resize_aspect_ratio


def test_net(model, image, device: str, score_thresh: float = 0.9, nms_thresh: float = 0.2, cover_thresh: float = 0.1, resize: int = 512):
    # resize
    img_resized, target_ratio, _ = resize_aspect_ratio(image, resize)
    ratio_h, ratio_w = 1 / target_ratio[0], 1 / target_ratio[1]

    # preprocessing
    img_resized = normalization(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        score, geo = model(img_resized)

    boxes = get_boxes(score.squeeze(0).cpu().numpy(),
                      geo.squeeze(0).cpu().numpy(),
                      score_thresh,
                      nms_thresh,
                      cover_thresh)
    
    return adjustResultCoordinates(np.asarray(boxes), ratio_w, ratio_h)


def test_onnx_net(model, image, score_thresh: float = 0.9, nms_thresh: float = 0.2, cover_thresh: float = 0.1, resize: int = 512):
    # resize
    img_resized, target_ratio, _ = resize_aspect_ratio(image, resize)
    ratio_h, ratio_w = 1 / target_ratio[0], 1 / target_ratio[1]

    # preprocessing
    img_resized = normalization(img_resized).numpy()
    img_resized = img_resized[np.newaxis, ...]

    # suy luận mô hình ONNX
    ort_inputs = {'input': img_resized}
    ort_outputs = model.run(None, ort_inputs)
    score, geo = ort_outputs[0], ort_outputs[1]

    boxes = get_boxes(score[0, ...],
                      geo[0, ...],
                      score_thresh,
                      nms_thresh,
                      cover_thresh)

    return adjustResultCoordinates(boxes, ratio_w, ratio_h)
