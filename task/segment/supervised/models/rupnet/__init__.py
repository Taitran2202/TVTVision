from task.segment.supervised.main import run
from .model import RUPNet


def build_model(cfg):
    model = RUPNet(
        feature_extractor_name=cfg['MODEL']['backbone']
    )

    return model
