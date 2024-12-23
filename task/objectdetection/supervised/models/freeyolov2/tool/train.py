import os
import json
import wandb
import time
import logging
from collections import OrderedDict
from utils.save_load import save_model, load_model
from utils.metrics.objectdetection import VOCAPIEvaluator
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
    loss_iou_m = AverageMeter()
    loss_cls_m = AverageMeter()
    loss_dfl_m = AverageMeter()
    losses_m = AverageMeter()

    end = time.time()

    model.train()
    optimizer.zero_grad()

    for idx, (inputs, targets) in enumerate(dataloader):
        data_time_m.update(time.time() - end)

        inputs = inputs.to(device)
        # predict
        outputs = model(inputs)

        loss_dict = criterion(outputs=outputs, targets=targets)
        loss_cls = loss_dict['loss_cls']
        loss_iou = loss_dict['loss_iou']
        loss_dfl = loss_dict['loss_dfl']
        loss = loss_dict['losses']
        loss.backward()

        # loss loss
        optimizer.step()
        optimizer.zero_grad()
        loss_iou_m.update(loss_iou.item())
        loss_cls_m.update(loss_cls.item())
        loss_dfl_m.update(loss_dfl.item())
        losses_m.update(loss.item())

        batch_time_m.update(time.time() - end)

        if idx % log_interval == 0 and idx != 0:
            _logger.info('TRAIN [{:>4d}/{}] '
                         'Loss: {loss.avg:>6.4f} '
                         'Loss IOU: {loss_iou.avg:>6.4f} '
                         'Loss Cls: {loss_cls.avg:>6.4f} '
                         'Loss Dfl: {loss_dfl.avg:>6.4f} '
                         'LR: {lr:.4e} '
                         'Time: {batch_time.avg:.3f}s, {rate:>7.2f}/s '
                         'Data: {data_time.avg:.3f}'.format(
                             idx+1, len(dataloader),
                             loss=losses_m,
                             loss_iou=loss_iou_m,
                             loss_cls=loss_cls_m,
                             loss_dfl=loss_dfl_m,
                             lr=optimizer.param_groups[0]['lr'],
                             batch_time=batch_time_m,
                             rate=inputs.size(0) / batch_time_m.val,
                             rate_avg=inputs.size(0) / batch_time_m.avg,
                             data_time=data_time_m))

        end = time.time()

    return OrderedDict([('loss', losses_m.avg)])


def test(model, validset, criterion, log_interval: int, vis_test_dir: str, device: str) -> dict:
    # Khởi tạo Evaluator
    evaluator = VOCAPIEvaluator()
    # Tính toán mAP
    mAP = evaluator.evaluate(
        model, validset, model.num_classes, vis_test_dir, device)
    _logger.info('Mean Average Precision (mAP): %.3f' % mAP)

    return OrderedDict([('mAP', mAP)])


def fit(
    model, trainloader, validset, criterion, optimizer, scheduler,
    epochs: int, savedir: str, log_interval: int, eval_interval: int,
    resume: bool, save_model_path: str, vis_test_dir: str, device: str
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
            eval_metrics = test(model, validset, criterion,
                                log_interval, vis_test_dir, device)
            metrics.update([('eval_' + k, v) for k, v in eval_metrics.items()])
            # checkpoint
            if best_score <= eval_metrics['mAP']:
                # save results
                state = {'best_epoch': epoch,
                         'best_score': eval_metrics['mAP']}
                json.dump(state, open(os.path.join(
                    savedir, f'best_score.json'), 'w'), indent=4)

                _logger.info('Best mAP {0:.3%} to {1:.3%}'.format(
                    best_score, eval_metrics['mAP']))

                best_score = eval_metrics['mAP']

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
