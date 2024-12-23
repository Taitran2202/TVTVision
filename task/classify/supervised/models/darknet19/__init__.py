from task.classify.supervised.main import run
from .model import build_darknet19


def build_model(cfg, cls):
    # build DarkNet19
    model = build_darknet19(
        model_name=cfg.MODEL.backbone,
        num_classes=len(cls),
        pretrained=cfg.MODEL.pretrained
    )
    return model
