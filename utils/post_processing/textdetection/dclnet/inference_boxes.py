import numpy as np
import torch
from .dclnet_utils import restore_bboxes, adjustResultCoordinates
from utils.pre_processing.imgproc import normalization, resize_aspect_ratio


def test_net(model, image, device: str, cls_thresh: float = 0.4, nms_thresh: float = 0.1, resize: int = 512):
    # resize
    img_resized, target_ratio, _ = resize_aspect_ratio(image, resize)
    ratio_h, ratio_w = 1 / target_ratio[0], 1 / target_ratio[1]

    # preprocessing
    img_resized = normalization(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        cls, rho, theta = model(img_resized)

    boxes = restore_bboxes(cls.squeeze(0).cpu().numpy(),
                           rho.squeeze(0).cpu().numpy(),
                           theta.squeeze(0).cpu().numpy(),
                           cls_thresh,
                           nms_thresh)
    return adjustResultCoordinates(boxes, ratio_w, ratio_h)


def test_onnx_net(model, image, cls_thresh: float = 0.4, nms_thresh: float = 0.2, cover_thresh: float = 0.1, resize: int = 512):
    # resize
    img_resized, target_ratio, _ = resize_aspect_ratio(image, resize)
    ratio_h, ratio_w = 1 / target_ratio[0], 1 / target_ratio[1]

    # preprocessing
    img_resized = normalization(img_resized).numpy()
    img_resized = img_resized[np.newaxis, ...]

    # suy luận mô hình ONNX
    ort_inputs = {'input': img_resized}
    ort_outputs = model.run(None, ort_inputs)
    cls, rho, theta = ort_outputs[0], ort_outputs[1], ort_outputs[2]

    boxes = restore_bboxes(cls[0, ...],
                           rho[0, ...],
                           theta[0, ...],
                           cls_thresh,
                           nms_thresh)
    return adjustResultCoordinates(boxes, ratio_w, ratio_h)
