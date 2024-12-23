import torch
from .sophia import SophiaG

optimizers = {
    'adam': (torch.optim.Adam, "Adam"),
    'adamw': (torch.optim.AdamW, "AdamW"),
    'sgd': (torch.optim.SGD, "SGD"),
    'sophiag': (SophiaG, "SophiaG"),
}


def build_optimizer(cfg, model):
    optimizer_type = None
    for optimizer_key, (optimizer_class, optimizer_name) in optimizers.items():
        if cfg['OPTIMIZER'][optimizer_key]['option']:
            optimizer_type = optimizer_class
            optimizer_name = optimizer_name
            break

    optimizer_config = cfg['OPTIMIZER'][optimizer_key]

    optimizer = optimizer_type(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=optimizer_config['lr'],
        weight_decay=optimizer_config['weight_decay'],
        **{k: v for k, v in optimizer_config.items() if k not in ['option', 'lr', 'weight_decay']}
    )

    return optimizer, optimizer_name, optimizer_config['lr']
