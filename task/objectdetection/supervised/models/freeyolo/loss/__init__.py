from .yolo_loss import YOLOLoss


def build_loss(cfg, device):
    criterion = YOLOLoss(
        center_sampling_radius=cfg.TRAIN.center_sampling_radius,
        topk_candicate=cfg.TRAIN.topk_candicate,
        device=device,
        loss_cls_weight=cfg.TRAIN.loss_cls_weight,
        loss_obj_weight=cfg.TRAIN.loss_obj_weight,
        loss_reg_weight=cfg.TRAIN.loss_reg_weight,
        num_classes=len(cfg.DATASET.class_names)
    )
    loss_name = "FreeYOLO Loss"
    return criterion, loss_name
