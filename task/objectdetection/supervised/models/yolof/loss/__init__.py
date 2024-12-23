from .yolo_loss import YOLOLoss


def build_loss(cfg, device):
    criterion = YOLOLoss(
        loss_cls_weight=cfg.TRAIN.loss_cls_weight,
        loss_reg_weight=cfg.TRAIN.loss_reg_weight,
        focal_loss_alpha=cfg.TRAIN.focal_loss_alpha,
        focal_loss_gamma=cfg.TRAIN.focal_loss_gamma,
        topk_candidates=cfg.TRAIN.topk_candidates,
        ignore_thresh=cfg.TRAIN.ignore_thresh,
        iou_thresh=cfg.TRAIN.iou_thresh,
        num_classes=len(cfg.DATASET.class_names)
    )
    loss_name = "YOLOF Loss"
    return criterion, loss_name
