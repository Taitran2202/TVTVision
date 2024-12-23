import os
import torch
from .model import CRAFT
from ...main import run


def build_model(cfg):
    # build CRAFT
    model = CRAFT(
        backbone=cfg.MODEL.backbone,
        pretrained=cfg.MODEL.pretrained
    )
    return model
