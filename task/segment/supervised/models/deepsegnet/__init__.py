from task.segment.supervised.main import run
from .model import DeepSegNet


def build_model(cfg):
    model = DeepSegNet(
        feature_extractor_name=cfg['MODEL']['feature_extractor_name']
    )
    return model
