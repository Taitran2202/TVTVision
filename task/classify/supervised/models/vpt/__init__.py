from task.classify.supervised.main import run
from .model import VPT


def build_model(cfg, cls):
    # build VPT
    model = VPT(
        backbone=cfg.MODEL.backbone,
        num_classes=len(cls),
        pretrained=cfg.MODEL.pretrained,
        prompt_tokens=cfg.MODEL.prompt_tokens,
        prompt_dropout=cfg.MODEL.prompt_dropout,
        prompt_type=cfg.MODEL.prompt_type
    )
    return model
