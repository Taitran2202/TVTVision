from task.classify.supervised.main import run
from .model import build_elan_cspnet


def build_model(cfg, cls):
    # build Elannet
    model = build_elan_cspnet(
        model_name=cfg.MODEL.backbone,
        num_classes=len(cls),
        pretrained=cfg.MODEL.pretrained
    )
    return model