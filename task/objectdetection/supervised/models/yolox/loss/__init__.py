from .yolo_loss import YOLOLoss


def build_loss(cfg, device):
    criterion = YOLOLoss(
        center_sampling_radius=cfg.TRAIN.center_sampling_radius,
        topk_candicate=cfg.TRAIN.topk_candicate,
        device=device,
        loss_cls_weight=cfg.TRAIN.loss_cls_weight,
        loss_obj_weight=cfg.TRAIN.loss_obj_weight,
        loss_box_weight=cfg.TRAIN.loss_box_weight,
        num_classes=len(cfg.DATASET.class_names),
        max_epoch=cfg.TRAIN.epochs,
        no_aug_epoch=cfg.TRAIN.no_aug_epoch
    )
    loss_name = "YOLOX Loss"
    return criterion, loss_name
