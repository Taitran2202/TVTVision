import numpy as np
import torch
from .utils import getDetBoxes, adjustResultCoordinates
from utils.pre_processing.imgproc import normalization, resize_aspect_ratio, cvt2HeatmapImg


def test_net(model, image, device: str, text_threshold: float, link_threshold: float, low_text: float, resize: int = 512):
    # resize
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
        image, resize
    )
    ratio_h, ratio_w = 1 / target_ratio[0], 1 / target_ratio[1]

    # preprocessing
    img_resized = normalization(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        region_score, affinity_score = model(img_resized)

    # make score and link map
    score_text = region_score[0, ...].cpu().numpy().astype(np.float32)
    score_link = affinity_score[0, ...].cpu().numpy().astype(np.float32)

    # NOTE
    score_text = score_text[: size_heatmap[0], : size_heatmap[1]]
    score_link = score_link[: size_heatmap[0], : size_heatmap[1]]

    # Post-processing
    boxes = getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text
    )

    # điều chỉnh tọa độ
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)

    render_score_text = cvt2HeatmapImg(score_text)
    render_score_link = cvt2HeatmapImg(score_link)
    render_img = [render_score_text, render_score_link]

    return boxes, render_img


def test_onnx_net(model, image, text_threshold: float, link_threshold: float, low_text: float, resize: int = 512):
    # resize
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
        image, resize)
    ratio_h, ratio_w = 1 / target_ratio[0], 1 / target_ratio[1]

    # preprocessing
    img_resized = normalization(img_resized).numpy()
    img_resized = img_resized[np.newaxis, ...]

    # suy luận mô hình ONNX
    ort_inputs = {'input': img_resized}
    ort_outputs = model.run(None, ort_inputs)
    region_score, affinity_score = ort_outputs[0], ort_outputs[1]

    # make score and link map
    score_text = region_score[0, ...].astype(np.float32)
    score_link = affinity_score[0, ...].astype(np.float32)

    # NOTE
    score_text = score_text[: size_heatmap[0], : size_heatmap[1]]
    score_link = score_link[: size_heatmap[0], : size_heatmap[1]]

    # Post-processing
    boxes = getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text
    )

    # điều chỉnh tọa độ
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)

    render_score_text = cvt2HeatmapImg(score_text)
    render_score_link = cvt2HeatmapImg(score_link)
    render_img = [render_score_text, render_score_link]

    return boxes, render_img
