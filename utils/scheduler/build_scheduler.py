import math
from functools import partial
import torch.optim.lr_scheduler as lr_scheduler
from .scheduler import CosineAnnealingWarmupRestarts


def custom_learning_rate_decay(epoch, total_epochs, initial_lr, scheduler_name):
    if scheduler_name == 'cosine':
        return ((1 - math.cos(epoch * math.pi / total_epochs)) / 2) * (initial_lr - 1) + 1

    elif scheduler_name == 'linear':
        return (1 - epoch / total_epochs) * (1.0 - initial_lr) + initial_lr

    elif scheduler_name == 'cos_linear':
        return (1 - epoch / total_epochs) * (1.0 - initial_lr) + initial_lr if epoch > total_epochs // 2 else ((1 - math.cos(epoch * math.pi / total_epochs)) / 2) * (initial_lr - 1) + 1


def build_scheduler(cfg, optimizer, lr):
    scheduler_type = None
    scheduler_config = None

    if cfg['SCHEDULER']['use_scheduler']:
        if cfg['SCHEDULER']['cosine_annealing_warmup_restarts']['option']:
            scheduler_type = CosineAnnealingWarmupRestarts
            scheduler_config = {
                'first_cycle_steps': cfg['TRAIN']['num_training_steps'],
                'max_lr': lr,
                'min_lr': cfg['SCHEDULER']['cosine_annealing_warmup_restarts']['min_lr'],
                'warmup_steps': int(cfg['TRAIN']['num_training_steps'] * cfg['SCHEDULER']['cosine_annealing_warmup_restarts']['warmup_ratio'])
            }
        elif cfg['SCHEDULER']['step_lr']['option']:
            scheduler_type = lr_scheduler.StepLR
            scheduler_config = {
                'step_size': int(0.95 * cfg['TRAIN']['num_training_steps']),
                'gamma': cfg['SCHEDULER']['step_lr']['gamma']
            }
        elif cfg['SCHEDULER']['cosine_annealing_lr']['option']:
            scheduler_type = lr_scheduler.CosineAnnealingLR
            scheduler_config = {
                'T_max': cfg['TRAIN']['epochs'],
                'eta_min': lr * 1e-2
            }
        elif cfg['SCHEDULER']['lambda_lr']['option']:
            scheduler_type = lr_scheduler.LambdaLR
            custom_lr_decay_func = partial(
                custom_learning_rate_decay,
                initial_lr=lr,
                total_epochs=cfg['TRAIN']['epochs'],
                scheduler_name=cfg['SCHEDULER']['lambda_lr']['scheduler_name']
            )
            scheduler_config = {
                'lr_lambda': custom_lr_decay_func
            }

    if scheduler_type is not None:
        scheduler = scheduler_type(optimizer, **scheduler_config)
    else:
        scheduler = None

    return scheduler
