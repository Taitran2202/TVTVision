import os
import torch
from ...main import run
from .model import RTTDet


def build_model(cfg):
    # build RTTDet
    model_type = cfg.MODEL.model_type
    model = RTTDet(
        cfg=cfg.MODEL[model_type],
        num_classes=2
    )
    return model
