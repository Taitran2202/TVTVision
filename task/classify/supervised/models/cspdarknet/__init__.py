from task.classify.supervised.main import run
from .model import build_cspdarknet


def build_model(cfg, cls):
    # build CSPDarkNet
    model = build_cspdarknet(
        model_name=cfg.MODEL.backbone,
        num_classes=len(cls),
        pretrained=cfg.MODEL.pretrained
    )
    return model
