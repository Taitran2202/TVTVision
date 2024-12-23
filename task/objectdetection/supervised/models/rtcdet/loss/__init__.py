from .yolo_loss import YOLOLoss


def build_loss(cfg, device):
    criterion = YOLOLoss(
        max_epoch=cfg.TRAIN.epochs,
        no_aug_epoch=cfg.TRAIN.no_aug_epoch,
        device=device,
        loss_cls_weight=cfg.TRAIN.loss_cls_weight,
        loss_iou_weight=cfg.TRAIN.loss_iou_weight,
        loss_dfl_weight=cfg.TRAIN.loss_dfl_weight,
        loss_box_aux=cfg.TRAIN.loss_box_aux,
        num_classes=len(cfg.DATASET.class_names),
        topk_candidate=cfg.TRAIN.topk_candidate,
        center_sampling_radius=cfg.TRAIN.center_sampling_radius,
        ema_update=cfg.TRAIN.ema_update,
        reg_max=16
    )
    loss_name = "RTCDet Loss"
    return criterion, loss_name
