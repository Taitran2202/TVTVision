from task.classify.supervised.main import run
from .model import build_convmixer


def build_model(cfg, cls):
    # build ConvMixer
    model = build_convmixer(
        model_name=cfg.MODEL.backbone,
        num_classes=len(cls),
        pretrained=cfg.MODEL.pretrained
    )
    return model
