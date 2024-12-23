from task.edgedetection.supervised.main import run
from .model import CTFN


def build_model(cfg):
    model = CTFN(
        fpn_name=cfg['MODEL']['fpn_name'],
        att_name=cfg['MODEL']['att_name']
    )
    return model
