from task.segment.supervised.main import run
from .model import ConvSeg


def build_model(cfg):
    model = ConvSeg()
    return model
