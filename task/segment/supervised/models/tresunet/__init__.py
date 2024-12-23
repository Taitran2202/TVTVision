from task.segment.supervised.main import run
from .model import TResUnet


def build_model(cfg):
    model = TResUnet(
        feature_extractor_name=cfg['MODEL']['backbone']
    )

    return model
