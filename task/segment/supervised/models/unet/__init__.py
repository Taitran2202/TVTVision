from task.segment.supervised.main import run
from .model import UNet


def build_model(cfg):
    model = UNet()
    return model
