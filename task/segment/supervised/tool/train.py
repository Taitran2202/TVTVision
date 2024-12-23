import os
import json
import wandb
import time
import logging
import numpy as np
from collections import OrderedDict
import torch
from utils.metrics.segment import calculate_metrics
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

    # metrics
    jac_m = AverageMeter()
    f1_m = AverageMeter()
    recall_m = AverageMeter()
    precision_m = AverageMeter()

    end = time.time()

    model.train()
    optimizer.zero_grad()

    for idx, (inputs, targets) in enumerate(dataloader):
        data_time_m.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)

        # predict
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # loss update
        optimizer.step()
        optimizer.zero_grad()
        losses_m.update(loss.item())

        """ Calculate the metrics """
        batch_jac = []
        batch_f1 = []
        batch_recall = []
        batch_precision = []

        for yt, yp in zip(targets, outputs):
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])

        # log loss
        losses_m.update(loss.item())
        jac_m.update(np.mean(batch_jac))
        f1_m.update(np.mean(batch_f1))
        recall_m.update(np.mean(batch_recall))
        precision_m.update(np.mean(batch_precision))

        batch_time_m.update(time.time() - end)

        if idx % log_interval == 0 and idx != 0:
            _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.avg:>6.4f} '
                         'Jaccard: {jac.avg:.3} '
                         'F1: {f1.avg:.3} '
                         'Recall: {recall.avg:.3} '
                         'Precision: {precision.avg:.3} '
                         'LR: {lr:.4e} '
                         'Time: {batch_time.avg:.3f}s, {rate:>7.2f}/s '
                         'Data: {data_time.avg:.3f}'.format(
                             idx+1, len(dataloader),
                             loss=losses_m,
                             jac=jac_m,
                             f1=f1_m,
                             recall=recall_m,
                             precision=precision_m,
                             lr=optimizer.param_groups[0]['lr'],
                             batch_time=batch_time_m,
                             rate=inputs.size(0) / batch_time_m.val,
                             rate_avg=inputs.size(0) / batch_time_m.avg,
                             data_time=data_time_m))

        end = time.time()

    return OrderedDict([('loss', losses_m.avg), ('jac', jac_m.avg), ('f1', f1_m.avg), ('recall', recall_m.avg), ('precision', precision_m.avg)])


def test(model, dataloader, criterion, log_interval: int, device: str) -> dict:
    losses_m = AverageMeter()

    # metrics
    jac_m = AverageMeter()
    f1_m = AverageMeter()
    recall_m = AverageMeter()
    precision_m = AverageMeter()

    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # predict
            outputs = model(inputs)

            # loss
            loss = criterion(outputs, targets)

            # total loss and acc
            """ Calculate the metrics """
            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []

            for yt, yp in zip(targets, outputs):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            # log loss
            losses_m.update(loss.item())
            jac_m.update(np.mean(batch_jac))
            f1_m.update(np.mean(batch_f1))
            recall_m.update(np.mean(batch_recall))
            precision_m.update(np.mean(batch_precision))

            if idx % log_interval == 0 and idx != 0:
                _logger.info('TEST [%d/%d]: Loss: %.3f | Jaccard: %.3f | F1: %.3f | Recall: %.3f | Precision: %.3f ' %
                             (idx+1, len(dataloader), losses_m.avg, jac_m.avg, f1_m.avg, recall_m.avg, precision_m.avg))

    return OrderedDict([('loss', losses_m.avg), ('jac', jac_m.avg), ('f1', f1_m.avg), ('recall', recall_m.avg), ('precision', precision_m.avg)])


def fit(
    model, trainloader, testloader, criterion, optimizer, scheduler,
    epochs: int, savedir: str, log_interval: int, eval_interval: int,
    resume: bool, save_model_path: str, device: str
) -> None:
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
                              optimizer, log_interval, device)
        # wandb
        metrics = OrderedDict(lr=optimizer.param_groups[0]['lr'])
        metrics.update([('train_' + k, v) for k, v in train_metrics.items()])

        if ((epoch+1) % eval_interval == 0) or (epoch+1) == epochs:
            eval_metrics = test(model, testloader,
                                criterion, log_interval, device)
            metrics.update([('eval_' + k, v) for k, v in eval_metrics.items()])
            # checkpoint
            if best_score <= eval_metrics['f1']:
                # save results
                state = {'best_epoch': epoch, 'best_score': eval_metrics['f1']}
                json.dump(state, open(os.path.join(
                    savedir, f'best_best_score.json'), 'w'), indent=4)

                _logger.info('Best F1 {0:.3%} to {1:.3%}'.format(
                    best_score, eval_metrics['f1']))

                best_score = eval_metrics['f1']

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

        wandb.log(metrics, step=step)

        step += 1

        # step scheduler
        if scheduler:
            scheduler.step()

    _logger.info('Best Metric: {0:.3%} (epoch {1:})'.format(
        state['best_score'], state['best_epoch']))
