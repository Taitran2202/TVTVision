import os
import torch
from ...main import run
from .model import RTTDetv2


def build_model(cfg):
    # build RTTDetv2
    model_type = cfg.MODEL.model_type
    model = RTTDetv2(
        cfg=cfg.MODEL[model_type],
        num_classes=2
    )
    return model
