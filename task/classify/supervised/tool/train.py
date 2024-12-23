import os
import numpy as np
import json
import wandb
import time
import logging
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, \
    balanced_accuracy_score, classification_report, confusion_matrix
import torch
from utils.metrics.classify import NoIndent
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


def accuracy(outputs, targets, return_correct=False):
    # calculate accuracy
    preds = outputs.argmax(dim=1)
    correct = targets.eq(preds).sum().item()

    if return_correct:
        return correct
    else:
        return correct/targets.size(0)


def calc_metrics(y_true: list, y_score: np.ndarray, y_pred: list, return_per_class: bool = False) -> dict:
    # softmax
    y_score = torch.nn.functional.softmax(torch.FloatTensor(y_score), dim=1)

    if y_score.shape[1] == 2:
        auroc = roc_auc_score(
            y_true, y_score[:, 1], average='macro', multi_class='ovr')
    else:
        auroc = roc_auc_score(
            y_true, y_score, average='macro', multi_class='ovr')

    # metrics
    f1 = f1_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    bcr = balanced_accuracy_score(y_true, y_pred)

    metrics = {
        'auroc': auroc,
        'f1': f1,
        'recall': recall,
        'precision': precision,
        'bcr': bcr
    }

    if return_per_class:
        # confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # merics per class
        f1_per_class = f1_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        precision_per_class = precision_score(y_true, y_pred, average=None)
        acc_per_class = cm.diagonal() / cm.sum(axis=1)

        metrics.update({
            'per_class': {
                'cm': [NoIndent(elem) for elem in cm.tolist()],
                'f1': f1_per_class.tolist(),
                'recall': recall_per_class.tolist(),
                'precision': precision_per_class.tolist(),
                'acc': acc_per_class.tolist()
            }
        })

    return metrics


def train(model, dataloader, criterion, optimizer, log_interval: int, device: str) -> dict:
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_m = AverageMeter()
    losses_m = AverageMeter()

    total_preds = []
    total_score = []
    total_targets = []

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

        # accuracy
        acc_m.update(accuracy(outputs, targets), n=targets.size(0))

        # stack output
        total_preds.extend(outputs.argmax(dim=1).detach().cpu().tolist())
        total_score.extend(outputs.detach().cpu().tolist())
        total_targets.extend(targets.detach().cpu().tolist())

        batch_time_m.update(time.time() - end)

        if (idx + 1) % log_interval == 0:
            _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                         'Acc: {acc.avg:.3%} '
                         'LR: {lr:.3e} '
                         'Time: {batch_time.avg:.3f}s, {rate:>7.2f}/s '
                         'Data: {data_time.avg:.3f}'.format(
                             idx+1, len(dataloader),
                             loss=losses_m,
                             acc=acc_m,
                             lr=optimizer.param_groups[0]['lr'],
                             batch_time=batch_time_m,
                             rate=inputs.size(0) / batch_time_m.val,
                             data_time=data_time_m))

        end = time.time()

    # calculate metrics
    metrics = calc_metrics(
        y_true=total_targets,
        y_score=total_score,
        y_pred=total_preds
    )

    metrics.update([('acc', acc_m.avg), ('loss', losses_m.avg)])

    # logging metrics
    _logger.info('\nTRAIN: Loss: %.3f | Acc: %.3f%% | BCR: %.3f%% | AUROC: %.3f%% | F1-Score: %.3f%% | Recall: %.3f%% | Precision: %.3f%%\n' %
                 (metrics['loss'], 100.*metrics['acc'], 100.*metrics['bcr'], 100.*metrics['auroc'], 100.*metrics['f1'], 100.*metrics['recall'], 100.*metrics['precision']))

    # classification report
    _logger.info(classification_report(
        y_true=total_targets, y_pred=total_preds, digits=4))

    return metrics


def test(model, dataloader, criterion, log_interval: int, device: str, return_per_class: bool = False) -> dict:
    correct = 0
    total = 0
    total_loss = 0

    total_preds = []
    total_score = []
    total_targets = []

    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # predict
            outputs = model(inputs)

            # loss
            loss = criterion(outputs, targets)

            # total loss and acc
            total_loss += loss.item()
            correct += accuracy(outputs, targets, return_correct=True)
            total += targets.size(0)

            # stack output
            total_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            total_score.extend(outputs.cpu().tolist())
            total_targets.extend(targets.cpu().tolist())

            if (idx + 1) % log_interval == 0:
                _logger.info('TEST [%d/%d]: Loss: %.3f | Acc: %.3f%% [%d/%d]' %
                             (idx+1, len(dataloader), total_loss/(idx+1), 100.*correct/total, correct, total))

    # calculate metrics
    metrics = calc_metrics(
        y_true=total_targets,
        y_score=total_score,
        y_pred=total_preds,
        return_per_class=return_per_class
    )

    metrics.update(
        [('acc', correct/total), ('loss', total_loss/len(dataloader))])

    # logging metrics
    _logger.info('\nTEST: Loss: %.3f | Acc: %.3f%% | BCR: %.3f%% | AUROC: %.3f%% | F1-Score: %.3f%% | Recall: %.3f%% | Precision: %.3f%%\n' %
                 (metrics['loss'], 100.*metrics['acc'], 100.*metrics['bcr'], 100.*metrics['auroc'], 100.*metrics['f1'], 100.*metrics['recall'], 100.*metrics['precision']))

    # classification report
    _logger.info(classification_report(
        y_true=total_targets, y_pred=total_preds, digits=4))

    return metrics


def fit(
    model, trainloader, testloader, criterion, optimizer, scheduler,
    epochs: int, savedir: str, log_interval: int, eval_interval: int,
    resume: bool, save_model_path: str, ckp_metric: str, device: str
) -> None:
    best_score = 0.
    step = 0

    # resume model
    if os.path.exists(os.path.join(savedir, save_model_path)) and resume:
        model, optimizer, scheduler, _, best_score = load_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            resume=os.path.join(savedir, save_model_path),
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
            ckp_cond = (best_score >= eval_metrics[ckp_metric]) if ckp_metric == 'loss' else (
                best_score <= eval_metrics[ckp_metric])
            metric_name = 'Loss' if ckp_metric == 'loss' else (
                'Accuracy' if ckp_metric == 'acc' else 'BCR')

            if ckp_cond:
                # save results
                state = {'best_epoch': epoch,
                         'best_score': eval_metrics[ckp_metric]}
                json.dump(state, open(os.path.join(
                    savedir, f'best_score.json'), 'w'), indent=4)

                _logger.info('Best {0} {1:.3%} to {2:.3%}'.format(
                    metric_name, best_score, eval_metrics[ckp_metric]))

                best_score = eval_metrics[ckp_metric]

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
