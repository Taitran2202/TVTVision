import os
import torch
from task.objectdetection.supervised.main import run
from .model import FreeYOLO


def build_model(cfg, device, size=None):
    # build FreeYOLO
    model_type = cfg.MODEL.model_type
    if size is not None:
        model_type = size
    model = FreeYOLO(
        cfg=cfg.MODEL[model_type],
        device=device,
        num_classes=len(cfg.DATASET.class_names),
        conf_thresh=cfg.TRAIN.conf_thresh,
        nms_thresh=cfg.TRAIN.nms_thresh,
        topk=cfg.TRAIN.topk,
        no_decode=cfg.TRAIN.no_decode
    )
    # Tải trọng số từ mô hình cũ
    checkpoint_path = os.path.join(
        cfg.TRAIN.base_weight, f'FreeYolo-{model_type}/best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        checkpoint_state_dict = checkpoint.pop("model")

        # Lọc và gán trọng số từ checkpoint vào mô hình mới
        model_state_dict = model.state_dict()
        for key in checkpoint_state_dict:
            if key in model_state_dict and not key.startswith("cls_preds"):
                model_state_dict[key] = checkpoint_state_dict[key]

        # Gán trọng số đã đồng nhất lại cho mô hình mới
        model.load_state_dict(model_state_dict, strict=False)
    return model


def build_onnx_model(cfg, device):
    # build FreeYOLO
    model = FreeYOLO(
        cfg=cfg.MODEL[cfg.MODEL.model_type],
        device=device,
        num_classes=len(cfg.DATASET.class_names),
        conf_thresh=cfg.TEST.conf_thresh,
        nms_thresh=cfg.TEST.nms_thresh,
        topk=cfg.TEST.topk,
        no_decode=cfg.TEST.no_decode
    )
    return model
