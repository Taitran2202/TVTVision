import os
import json
import numpy as np
import wandb
import time
import logging
import torch
from collections import OrderedDict
from ..util import save_result
from utils.save_load import save_model, load_model
from utils.metrics.edgedetection import compute_ods
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


def train(model, dataloader, criterion, optimizer, l_weight, log_interval: int, vis_train_dir: str, device: str) -> dict:
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    ods_m = AverageMeter()

    end = time.time()

    model.train()
    optimizer.zero_grad()

    for idx, (inputs, targets, img_names) in enumerate(dataloader):
        data_time_m.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)

        # predict
        outputs_list = model(inputs)
        ods = compute_ods(outputs_list[-1], targets)

        # save result
        save_result(outputs_list, vis_train_dir, img_names)

        # Loss
        loss = sum([criterion(preds, targets, l_w, device)
                   for preds, l_w in zip(outputs_list, l_weight)])

        loss.backward()

        # loss update
        optimizer.step()
        optimizer.zero_grad()
        losses_m.update(loss.item())
        ods_m.update(ods.item())

        batch_time_m.update(time.time() - end)

        if (idx + 1) % log_interval == 0:
            _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.avg:>6.4f} '
                         'ODS: {ods.avg:.3%} '
                         'LR: {lr:.3e} '
                         'Time: {batch_time.avg:.3f}s, {rate:>7.2f}/s '
                         'Data: {data_time.avg:.3f}'.format(
                             idx+1, len(dataloader),
                             loss=losses_m,
                             ods=ods_m,
                             lr=optimizer.param_groups[0]['lr'],
                             batch_time=batch_time_m,
                             rate=inputs.size(0) / batch_time_m.val,
                             data_time=data_time_m))

        end = time.time()

    return OrderedDict([('loss', losses_m.avg), ('ods', losses_m.avg)])


def test(model, dataloader, criterion, l_weight, log_interval: int, vis_test_dir: str, device: str) -> dict:
    total_loss = 0
    ods_m = AverageMeter()

    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets, img_names) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # predict
            outputs_list = model(inputs)
            ods = compute_ods(outputs_list[-1], targets)

            # save result
            save_result(outputs_list, vis_test_dir, img_names)

            # Loss
            loss = sum([criterion(preds, targets, l_w, device)
                        for preds, l_w in zip(outputs_list, l_weight)])

            # total loss
            total_loss += loss.item()
            ods_m.update(ods.item())

            if (idx + 1) % log_interval == 0:
                _logger.info('TEST [%d/%d]: Loss: %.3f ODS: %.3f%%' %
                             (idx+1, len(dataloader), total_loss/(idx+1), 100.*ods_m.avg))

    return OrderedDict([('loss', total_loss/len(dataloader)), ('ods', ods_m.avg)])


def fit(
    model, trainloader, testloader, criterion, optimizer, scheduler, l_weight,
    epochs: int, savedir: str, log_interval: int, resume: bool, save_model_path: str,
    vis_train_dir: str, vis_test_dir: str, device: str
) -> None:
    # best_score = float("inf")
    best_score = 0.
    step = 0

    # resume model
    if os.path.exists(os.path.join(savedir, save_model_path)) and resume:
        model, optimizer, scheduler, _, best_score = load_model(
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
                              optimizer, l_weight, log_interval, vis_train_dir, device)
        eval_metrics = test(model, testloader, criterion,
                            l_weight, log_interval, vis_test_dir, device)

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
        # if best_score >= eval_metrics['loss']:
        if best_score <= eval_metrics['ods']:
            # save results
            state = {'best_epoch': epoch, 'best_score': eval_metrics['ods']}
            json.dump(state, open(os.path.join(
                savedir, f'best_score.json'), 'w'), indent=4)

            _logger.info('Best ods {0:.3%} to {1:.3%}'.format(
                best_score, eval_metrics['ods']))

            best_score = eval_metrics['ods']

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

    _logger.info('Best Metric: {0:.3%} (epoch {1:})'.format(
        state['best_score'], state['best_epoch']))
