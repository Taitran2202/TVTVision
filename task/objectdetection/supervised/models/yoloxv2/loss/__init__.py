from .yolo_loss import YOLOLoss


def build_loss(cfg, device):
    criterion = YOLOLoss(
        soft_center_radius=cfg.TRAIN.soft_center_radius,
        topk_candidates=cfg.TRAIN.topk_candidates,
        device=device,
        loss_cls_weight=cfg.TRAIN.loss_cls_weight,
        loss_box_weight=cfg.TRAIN.loss_box_weight,
        num_classes=len(cfg.DATASET.class_names),
        max_epoch=cfg.TRAIN.epochs,
        no_aug_epoch=cfg.TRAIN.no_aug_epoch
    )
    loss_name = "YOLOXv2 Loss"
    return criterion, loss_name
