from task.edgedetection.supervised.main import run
from .model import LDC


def build_model(cfg):
    model = LDC()
    return model
