import os
import json
import wandb
import time
import logging
import cv2
from typing import List
from collections import OrderedDict
import torch
from utils.metrics.textdetection import DetectionIoUEvaluator
from utils.post_processing.textdetection.pan import test_net
from utils.save_load import save_model, load_model
from ..util import save_result_icdar13, save_result_icdar1517, save_result_synth
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


def train(model, dataloader, criterion, optimizer, loss_weights: List[float], log_interval: int, device: str) -> dict:
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    losses_text_m = AverageMeter()
    losses_kernels_m = AverageMeter()
    losses_emb_m = AverageMeter()
    ious_text_m = AverageMeter()
    ious_kernel_m = AverageMeter()

    # criterion
    pan_criterion = criterion
    text_weight, kernel_weight, emb_weight = loss_weights

    end = time.time()

    # set train mode
    model.train()

    # set optimizer
    optimizer.zero_grad()

    for idx, data in enumerate(dataloader):
        images = data['imgs'].to(device)
        gt_texts = data['gt_texts'].to(device)
        gt_kernels = data['gt_kernels'].to(device)
        training_masks = data['training_masks'].to(device)
        gt_instances = data['gt_instances'].to(device)
        gt_bboxes = data['gt_bboxes'].to(device)

        data_time_m.update(time.time() - end)

        # predict
        det_out = model(images)
        det_loss = pan_criterion(det_out, gt_texts, gt_kernels, training_masks,
                                 gt_instances, gt_bboxes, text_weight, kernel_weight, emb_weight)

        # detection loss
        loss_text = torch.mean(det_loss['loss_text'])
        loss_kernels = torch.mean(det_loss['loss_kernels'])
        loss_emb = torch.mean(det_loss['loss_emb'])
        iou_text = torch.mean(det_loss['iou_text'])
        iou_kernel = torch.mean(det_loss['iou_kernel'])

        loss = loss_text + loss_kernels + loss_emb
        loss.backward()

        # update weight
        optimizer.step()
        optimizer.zero_grad()

        # log loss
        losses_m.update(loss.item())
        losses_text_m.update(loss_text.item())
        losses_kernels_m.update(loss_kernels.item())
        losses_emb_m.update(loss_emb.item())
        ious_text_m.update(iou_text.item())
        ious_kernel_m.update(iou_kernel.item())

        batch_time_m.update(time.time() - end)

        if (idx+1) % log_interval == 0:
            _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                         'Text Loss: {text_loss.avg:>6.4f} '
                         'Kernel Loss: {kernel_loss.avg:>6.4f} '
                         'Emb Loss: {emb_loss.avg:>6.4f} '
                         'Text IoU: {text_iou.avg:>6.4f} '
                         'Kernel IoU: {kernel_iou.avg:>6.4f} '
                         'LR: {lr:.3e} '
                         'Time: {batch_time.avg:.3f}s, {rate:>7.2f}/s '
                         'Data: {data_time.avg:.3f}'.format(
                             idx+1, len(dataloader),
                             loss=losses_m,
                             text_loss=losses_text_m,
                             kernel_loss=losses_kernels_m,
                             emb_loss=losses_emb_m,
                             text_iou=ious_text_m,
                             kernel_iou=ious_kernel_m,
                             lr=optimizer.param_groups[0]['lr'],
                             batch_time=batch_time_m,
                             rate=images.size(0) / batch_time_m.val,
                             data_time=data_time_m))

        end = time.time()

    return OrderedDict([('loss', losses_m.avg), ('text loss', losses_text_m.avg), ('kernel loss', losses_kernels_m.avg), ('emb loss', losses_emb_m.avg)])


def test(model, dataset, criterion, min_area: float, min_score: float,
         bbox_type: str, resize: int, vis_test_dir: str, log_interval: int,
         device: str) -> dict:
    # metrics
    evaluator = DetectionIoUEvaluator()
    model.eval()
    total_imgs_bboxes_gt = []
    total_imgs_bboxes_pre = []

    with torch.no_grad():
        for _, (img_path, single_img_bboxes) in enumerate(dataset):
            total_imgs_bboxes_gt.append(single_img_bboxes)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            single_img_bbox = []
            bboxes, _ = test_net(
                model=model,
                image=image,
                device=device,
                min_area=min_area,
                min_score=min_score,
                bbox_type=bbox_type,
                resize=resize
            )

            for box in bboxes:
                box_info = {"points": box, "text": "###", "ignore": False}
                single_img_bbox.append(box_info)

            total_imgs_bboxes_pre.append(single_img_bbox)

            if 'icdar15' in vis_test_dir:
                save_result_icdar1517(
                    img_path=img_path,
                    img=image[:, :, ::-1].copy(),
                    pre_box=bboxes,
                    gt_box=single_img_bboxes,
                    result_dir=vis_test_dir
                )
            elif 'icdar13' in vis_test_dir:
                save_result_icdar13(
                    img_path=img_path,
                    img=image[:, :, ::-1].copy(),
                    pre_box=bboxes,
                    gt_box=single_img_bboxes,
                    result_dir=vis_test_dir
                )
            elif 'icdar17' in vis_test_dir:
                save_result_icdar1517(
                    img_path=img_path,
                    img=image[:, :, ::-1].copy(),
                    pre_box=bboxes,
                    gt_box=single_img_bboxes,
                    result_dir=vis_test_dir
                )
            elif 'synth' in vis_test_dir:
                save_result_synth(
                    img_path=img_path,
                    img=image[:, :, ::-1].copy(),
                    pre_box=bboxes,
                    gt_box=single_img_bboxes,
                    result_dir=vis_test_dir
                )

    results = []
    for _, (gt, pred) in enumerate(zip(total_imgs_bboxes_gt, total_imgs_bboxes_pre)):
        perSampleMetrics_dict = evaluator.evaluate_image(gt, pred)
        results.append(perSampleMetrics_dict)

    metrics = evaluator.combine_results(results)

    _logger.info('TEST: Hmean: {0:.3%} | Recall: {1:.3%} | Precision: {2:.3%}'.format
                 (metrics['hmean'], metrics['recall'], metrics['precision']))

    return metrics


def fit(
    model, trainloader, testdataset, criterion, optimizer, scheduler,
    loss_weights: List[float], min_area: float, min_score: float,
    bbox_type: str, resize: int, epochs: int, savedir: str,
    log_interval: int, eval_interval: int, resume: bool,
    save_model_path: str, vis_test_dir: str, device: str
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
            optimizer=optimizer,
            loss_weights=loss_weights,
            log_interval=log_interval,
            device=device
        )
        # wandb
        metrics = OrderedDict(lr=optimizer.param_groups[0]['lr'])
        metrics.update([('train_' + k, v) for k, v in train_metrics.items()])

        if ((epoch+1) % eval_interval == 0) or (epoch+1) == epochs:
            eval_metrics = test(
                model=model,
                dataset=testdataset,
                criterion=criterion,
                min_area=min_area,
                min_score=min_score,
                bbox_type=bbox_type,
                resize=resize,
                vis_test_dir=vis_test_dir,
                log_interval=log_interval,
                device=device
            )
            metrics.update([('eval_' + k, v) for k, v in eval_metrics.items()])

            # checkpoint
            if best_score <= eval_metrics['hmean']:
                # save results
                state = {'best_epoch': epoch,
                         'best_score': eval_metrics['hmean']}
                json.dump(state, open(os.path.join(
                    savedir, f'best_results.json'), 'w'), indent=4)

                _logger.info('Best Hmean {0:.3%} to {1:.3%}'.format(
                    best_score, eval_metrics['hmean']))

                best_score = eval_metrics['hmean']

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

    _logger.info('Best Hmean: {0:.3%} (epoch {1:})'.format(
        state['best_score'], state['best_epoch']))
