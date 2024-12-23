from task.edgedetection.supervised.main import run
from .model import CATS


def build_model(cfg):
    model = CATS()
    return model
