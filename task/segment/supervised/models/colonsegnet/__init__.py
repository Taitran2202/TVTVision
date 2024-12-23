from task.segment.supervised.main import run
from .model import ColonSegNet


def build_model(cfg):
    model = ColonSegNet()
    return model
