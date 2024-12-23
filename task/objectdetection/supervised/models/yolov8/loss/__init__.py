from .yolo_loss import YOLOLoss


def build_loss(cfg, device):
    model_type = cfg.MODEL.model_type
    criterion = YOLOLoss(
        reg_max=cfg.MODEL[model_type].reg_max,
        loss_cls_weight=cfg.TRAIN.loss_cls_weight,
        loss_box_weight=cfg.TRAIN.loss_box_weight,
        loss_dfl_weight=cfg.TRAIN.loss_dfl_weight,
        topk_candidates=cfg.TRAIN.topk_candidates,
        alpha=cfg.TRAIN.alpha,
        beta=cfg.TRAIN.beta,
        num_classes=len(cfg.DATASET.class_names)
    )
    loss_name = "YOLOv8 Loss"
    return criterion, loss_name
