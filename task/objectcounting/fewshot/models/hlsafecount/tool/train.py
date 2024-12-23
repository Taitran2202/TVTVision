import os
import json
import wandb
import time
import logging
import torch
from collections import OrderedDict
from utils.metrics.objectcounting import performances
from utils.save_load import save_model, load_model
_logger = logging.getLogger('train')


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(model, dataloader, criterion, optimizer, log_interval: int, device: str) -> dict:
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    end = time.time()

    model.train()
    optimizer.zero_grad()

    for idx, (inputs, density_gt, boxes, _) in enumerate(dataloader):
        data_time_m.update(time.time() - end)

        inputs, density_gt, boxes = inputs.to(
            device), density_gt.to(device), boxes.to(device)

        # predict
        density_pred, _ = model(inputs)
        loss = criterion(density_pred, density_gt)
        loss.backward()

        # loss update
        optimizer.step()
        optimizer.zero_grad()
        losses_m.update(loss.item())

        batch_time_m.update(time.time() - end)

        if (idx + 1) % log_interval == 0:
            _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.avg:>6.4f} '
                         'LR: {lr:.3e} '
                         'Time: {batch_time.avg:.3f}s, {rate:>7.2f}/s '
                         'Data: {data_time.avg:.3f}'.format(
                             idx+1, len(dataloader),
                             loss=losses_m,
                             lr=optimizer.param_groups[0]['lr'],
                             batch_time=batch_time_m,
                             rate=inputs.size(0) / batch_time_m.val,
                             data_time=data_time_m))

        end = time.time()

    return OrderedDict([('loss', losses_m.avg)])


def test(model, dataloader, criterion, log_interval: int, device: str) -> dict:
    mae_metric = AverageMeter()
    mse_metric = AverageMeter()

    model.eval()
    with torch.no_grad():
        for idx, (inputs, density_gt, boxes, cnts_gt) in enumerate(dataloader):
            inputs, density_gt, boxes, cnts_gt = inputs.to(device), density_gt.to(
                device), boxes.to(device), cnts_gt.to(device)

            # predict
            _, peakmaps = model(inputs)

            # update metrics
            mae_value, mse_value = performances(cnts_gt, peakmaps)
            mae_metric.update(mae_value.item())
            mse_metric.update(mse_value.item())

            if (idx + 1) % log_interval == 0:
                _logger.info('TEST [%d/%d]: MAE: %.1f | MSE: %.1f ' %
                             (idx+1, len(dataloader), mae_metric.avg, mse_metric.avg))

    return OrderedDict([('mae', mae_metric.avg), ('mse', mse_metric.avg)])


def fit(
    model, trainloader, testloader, criterion, optimizer, scheduler, epochs: int, savedir: str,
    log_interval: int, resume: bool = False, save_model_path: str = None, device: str = None
) -> None:
    best_score = float("inf")
    step = 0

    # resume model
    if os.path.exists(os.path.join(savedir, save_model_path)) and resume:
        model, optimizer, scheduler, best_score = load_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            resume=os.path.join(
                savedir, save_model_path),
            device=device,
            save_step=False
        )

    state = {'best_epoch': step+1, 'best_score': best_score}

    for epoch in range(epochs):
        _logger.info(f'\nEpoch: {epoch+1}/{epochs}')
        train_metrics = train(model, trainloader, criterion,
                              optimizer, log_interval, device)
        eval_metrics = test(model, testloader, criterion, log_interval, device)

        # wandb
        metrics = OrderedDict(lr=optimizer.param_groups[0]['lr'])
        metrics.update([('train_' + k, v) for k, v in train_metrics.items()])
        metrics.update([('eval_' + k, v) for k, v in eval_metrics.items()])
        wandb.log(metrics, step=step)

        step += 1

        # step scheduler
        if scheduler:
            scheduler.step()

        # checkpoint
        if best_score >= eval_metrics['mae']:
            # save results
            state = {'best_epoch': epoch, 'best_score': eval_metrics['mae']}
            json.dump(state, open(os.path.join(
                savedir, f'best_results.json'), 'w'), indent=4)

            _logger.info('Best MAE %.1f to %.1f' % (
                best_score, eval_metrics['mae']))

            best_score = eval_metrics['mae']

            # save model
            save_model(
                path=os.path.join(savedir, save_model_path),
                step=None,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_score=best_score,
                save_step=False
            )

    _logger.info('Best Metric: %.1f (epoch %d)' % (
        state['best_score'], state['best_epoch']))
