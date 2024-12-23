from task.classify.supervised.main import run
from .model import build_torchvision_model


def build_model(cfg, cls):
    # build MyModel
    model = build_torchvision_model(
        network=cfg.MODEL.backbone,
        num_classes=len(cls),
        pretrained=cfg.MODEL.pretrained
    )
    return model
