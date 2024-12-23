import torch
import logging
_logger = logging.getLogger('train')


def save_model(path, step, model, optimizer, scheduler, best_score, save_step=True):
    state_dict = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'best_score': best_score,
    }

    if not save_step:
        del state_dict['step']

    torch.save(state_dict, path)


def load_model(model, optimizer, scheduler, resume, device, save_step=True):
    checkpoint = torch.load(resume, map_location=device)
    _logger.info('loaded weights from {}, {}'.format(
        resume, 'step {}, '.format(checkpoint['step']) if save_step else '') +
        'best_score {}'.format(checkpoint['best_score']))

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    step = checkpoint['step'] if save_step else None
    best_score = checkpoint['best_score']

    return model, optimizer, scheduler, step, best_score
