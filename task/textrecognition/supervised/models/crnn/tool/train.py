import os
import json
import wandb
import time
import logging
from typing import List
from collections import OrderedDict
import torch
from utils.post_processing.textrecognition.crnn import ctc_decode
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


def train(model, dataloader, criterion, decode_method, beam_size, optimizer, log_interval: int, device: str) -> dict:
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    end = time.time()

    correct = 0
    total = 0

    model.train()
    optimizer.zero_grad()
    for idx, (inputs, targets, target_lengths) in enumerate(dataloader):
        data_time_m.update(time.time() - end)

        inputs, targets, target_lengths = inputs.to(
            device, dtype=torch.float32), targets.to(device), target_lengths.to(device)

        # predict
        log_probs = model(inputs)

        # loss
        batch_size = inputs.size(0)
        inputLengths = torch.LongTensor([log_probs.size(0)] * batch_size)
        targetLengths = torch.flatten(target_lengths)
        loss = criterion(log_probs, targets, inputLengths, targetLengths)

        # pedict
        preds = ctc_decode(log_probs.cpu().detach().numpy(),
                           method=decode_method, beam_size=beam_size)
        reals = targets.cpu().numpy().tolist()
        target_lengths = target_lengths.cpu().numpy().tolist()

        # total acc
        target_length_counter = 0
        for y_pred, target_length in zip(preds, target_lengths):
            real = reals[target_length_counter:target_length_counter + target_length]
            target_length_counter += target_length

            if y_pred == real:
                correct += 1

        total += batch_size

        loss.backward()

        # loss update
        optimizer.step()
        optimizer.zero_grad()

        # log loss
        losses_m.update(loss.item())
        batch_time_m.update(time.time() - end)

        if (idx + 1) % log_interval == 0:
            _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                         'Acc: {acc:.3%} [{correct:d}/{total:d}] '
                         'LR: {lr:.3e} '
                         'Time: {batch_time.avg:.3f}s, {rate:>7.2f}/s '
                         'Data: {data_time.avg:.3f}'.format(
                             idx+1, len(dataloader),
                             loss=losses_m,
                             acc=100.*correct/total,
                             correct=correct,
                             total=total,
                             lr=optimizer.param_groups[0]['lr'],
                             batch_time=batch_time_m,
                             rate=inputs.size(0) / batch_time_m.val,
                             data_time=data_time_m))

        end = time.time()

    metrics = {}
    metrics.update([('loss', losses_m.avg), ('acc', correct/total)])

    # logging metrics
    _logger.info('\nTRAIN: Loss: %.3f | Acc: %.3f%% \n' % (metrics['loss'], 100.*metrics['acc']))

    return metrics


def test(model, dataloader, criterion, decode_method, beam_size, log_interval, device: str = 'cpu') -> dict:
    correct = 0
    total = 0
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets, target_lengths) in enumerate(dataloader):
            inputs, targets, target_lengths = inputs.to(
                device, dtype=torch.float32), targets.to(device), target_lengths.to(device)

            # predict
            log_probs = model(inputs)

            # loss
            batch_size = inputs.size(0)
            inputLengths = torch.LongTensor([log_probs.size(0)] * batch_size)
            targetLengths = torch.flatten(target_lengths)
            loss = criterion(log_probs, targets, inputLengths, targetLengths)
            total_loss += loss.item()

            # pedict
            preds = ctc_decode(log_probs.cpu().detach().numpy(
            ), method=decode_method, beam_size=beam_size)
            reals = targets.cpu().numpy().tolist()
            target_lengths = target_lengths.cpu().numpy().tolist()

            # total acc
            target_length_counter = 0
            for y_pred, target_length in zip(preds, target_lengths):
                real = reals[target_length_counter:target_length_counter + target_length]
                target_length_counter += target_length

                if y_pred == real:
                    correct += 1

            total += inputs.size(0)

            # stack output
            if (idx + 1) % log_interval == 0:
                _logger.info('TEST [%d/%d]: Loss: %.3f | Acc: %.3f%% [%d/%d]' %
                             (idx+1, len(dataloader), total_loss/(idx+1), 100.*correct/total, correct, total))

    metrics = {}
    metrics.update(
        [('loss', total_loss/len(dataloader)), ('acc', correct/total)])

    # logging metrics
    _logger.info('\nTEST: Loss: %.3f | Acc: %.3f%% \n' %
                 (metrics['loss'], 100.*metrics['acc']))

    return metrics


def fit(
    model, trainloader, testloader, criterion, optimizer, scheduler,
    epochs: int, loss_weights: List[float], log_interval: int = 1,
    eval_interval: int = 1, savedir: str = None, resume: bool = None,
    save_model_path: str = None, decode_method: str = None, beam_size: int = None,
    use_wandb: bool = False, device: str = 'cpu'
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
        train_metrics = train(
            model=model,
            dataloader=trainloader,
            criterion=criterion,
            decode_method=decode_method,
            beam_size=beam_size,
            optimizer=optimizer,
            log_interval=log_interval,
            device=device,
        )

        # wandb
        metrics = OrderedDict(lr=optimizer.param_groups[0]['lr'])
        metrics.update([('train_' + k, v) for k, v in train_metrics.items()])

        if ((epoch+1) % eval_interval == 0) or (epoch+1) == epochs:
            eval_metrics = test(
                model=model,
                dataloader=testloader,
                criterion=criterion,
                decode_method=decode_method,
                beam_size=beam_size,
                log_interval=log_interval,
                device=device
            )
            metrics.update([('eval_' + k, v) for k, v in eval_metrics.items()])

            # checkpoint
            if best_score <= eval_metrics['acc']:
                # save results
                state = {'best_epoch': epoch,
                         'best_score': eval_metrics['acc']}
                json.dump(state, open(os.path.join(
                    savedir, f'best_score.json'), 'w'), indent=4)

                _logger.info('Best {0} {1:.3%} to {2:.3%}'.format(
                    "Accuracy", best_score, eval_metrics['acc']))

                best_score = eval_metrics['acc']

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
