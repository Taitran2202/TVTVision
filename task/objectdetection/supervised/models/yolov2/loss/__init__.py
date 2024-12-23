from .yolo_loss import YOLOLoss


def build_loss(cfg, device):
    criterion = YOLOLoss(
        device=device,
        loss_cls_weight=cfg.TRAIN.loss_cls_weight,
        loss_obj_weight=cfg.TRAIN.loss_obj_weight,
        loss_box_weight=cfg.TRAIN.loss_box_weight,
        iou_thresh=cfg.TRAIN.iou_thresh,
        anchor_size=cfg.MODEL.anchor_size,
        num_classes=len(cfg.DATASET.class_names),
    )
    loss_name = "YOLOv2 Loss"
    return criterion, loss_name
