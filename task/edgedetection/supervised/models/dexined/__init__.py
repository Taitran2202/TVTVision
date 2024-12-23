from task.edgedetection.supervised.main import run
from .model import DexiNed


def build_model(cfg):
    model = DexiNed()
    return model
