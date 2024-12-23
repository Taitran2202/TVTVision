from .yolo_loss import YOLOLoss


def build_loss(cfg, device):
    model_type = cfg.MODEL.model_type
    criterion = YOLOLoss(
        reg_max=cfg.MODEL[model_type].reg_max,
        cls_loss_name=cfg.TRAIN.cls_loss_name,
        device=device,
        loss_cls_weight=cfg.TRAIN.loss_cls_weight,
        loss_iou_weight=cfg.TRAIN.loss_iou_weight,
        loss_dfl_weight=cfg.TRAIN.loss_dfl_weight,
        num_classes=len(cfg.DATASET.class_names),
        topk=cfg.TRAIN.topk,
        alpha=cfg.TRAIN.alpha,
        beta=cfg.TRAIN.beta
    )
    loss_name = "FreeYOLOv2 Loss"
    return criterion, loss_name
