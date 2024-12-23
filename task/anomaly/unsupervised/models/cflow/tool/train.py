import os
import json
import numpy as np
import wandb
import time
import einops
import logging
from tqdm import tqdm
from typing import List
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from ..model.module import get_logp, positional_encoding_2d
from utils.metrics.anomaly import AnomalyScoreThreshold, compute_pro, trapezoid
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


def training(model, trainloader, validloader, criterion, optimizer, scheduler, num_training_steps: int = 1000, loss_weights: List[float] = [0.6, 0.4],
             log_interval: int = 1, eval_interval: int = 1, savedir: str = None, resume: bool = False, save_model_path: str = None, use_wandb: bool = False,
             top_k: int = 100, compute_threshold: bool = False, beta: float = 1.0, device: str = 'cpu') -> dict:
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    # metrics
    image_threshold = AnomalyScoreThreshold(
        default_value=0.5, beta=beta).to(device)
    pixel_threshold = AnomalyScoreThreshold(
        default_value=0.5, beta=beta).to(device)

    # training
    best_score = 0
    step = 0
    train_mode = True

    # resume model
    if os.path.exists(os.path.join(savedir, save_model_path)) and resume:
        model, optimizer, scheduler, step, best_score = load_model(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            resume=os.path.join(savedir, save_model_path),
            device=device
        )

        state = {'best_step': step}

    # set train mode
    model.train()

    # set optimizer
    optimizer.zero_grad()

    while train_mode:

        end = time.time()

        for inputs, targets in trainloader:
            inputs, masks, labels = inputs.to(device, dtype=torch.float32), targets['mask'].to(
                device), targets['label'].to(device)

            data_time_m.update(time.time() - end)

            # predict
            model.encoder.eval()
            activation = model.encoder(model.normalize(inputs))
            avg_loss = torch.zeros([1], dtype=torch.float64).to(device)
            height = []
            width = []
            for layer_idx, layer in enumerate(model.pool_layers):
                encoder_activations = activation[layer].detach()  # BxCxHxW
                batch_size, dim_feature_vector, im_height, im_width = encoder_activations.size()
                image_size = im_height * im_width
                # number of rows in the conditional vector
                embedding_length = batch_size * image_size

                height.append(im_height)
                width.append(im_width)
                # repeats positional encoding for the entire batch 1 C H W to B C H W
                pos_encoding = einops.repeat(
                    positional_encoding_2d(
                        model.condition_vector, im_height, im_width).unsqueeze(0),
                    "b c h w-> (tile b) c h w",
                    tile=batch_size,
                ).to(device)
                c_r = einops.rearrange(
                    pos_encoding, "b c h w -> (b h w) c")  # BHWxP
                e_r = einops.rearrange(
                    encoder_activations, "b c h w -> (b h w) c")  # BHWxC
                perm = torch.randperm(embedding_length)  # BHW
                decoder = model.decoders[layer_idx].to(device)

                fiber_batches = embedding_length // model.fiber_batch_size  # number of fiber batches
                assert fiber_batches > 0, "Make sure we have enough fibers, otherwise decrease N or batch-size!"

                for batch_num in range(fiber_batches):  # per-fiber processing
                    optimizer.zero_grad()
                    if batch_num < (fiber_batches - 1):
                        idx = torch.arange(
                            batch_num *
                            model.fiber_batch_size, (batch_num + 1) *
                            model.fiber_batch_size
                        )
                    else:  # When non-full batch is encountered batch_num * N will go out of bounds
                        idx = torch.arange(
                            batch_num * model.fiber_batch_size, embedding_length)
                    # get random vectors
                    c_p = c_r[perm[idx]]  # NxP
                    e_p = e_r[perm[idx]]  # NxC
                    # decoder returns the transformed variable z and the log Jacobian determinant
                    p_u, log_jac_det = decoder(e_p, [c_p])
                    #
                    decoder_log_prob = get_logp(
                        dim_feature_vector, p_u, log_jac_det)
                    log_prob = decoder_log_prob / dim_feature_vector  # likelihood per dim
                    loss = -F.logsigmoid(log_prob).mean()
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss.sum()

            # log loss
            losses_m.update(avg_loss.item())

            batch_time_m.update(time.time() - end)

            # wandb
            if use_wandb:
                wandb.log({
                    'lr': optimizer.param_groups[0]['lr'],
                    'train_loss': losses_m.val
                },
                    step=step)

            if (step+1) % log_interval == 0 or step == 0:
                _logger.info(
                    'TRAIN [{:>4d}/{}] '
                    'Loss: {loss.avg:>6.4f} '
                    'LR: {lr:.3e} '
                    'Time: {rate_avg:>7.2f}/s '
                    'Data: {data_time.avg:.3f}'.format(
                        step + 1, num_training_steps,
                        loss=losses_m,
                        lr=optimizer.param_groups[0]['lr'],
                        rate_avg=inputs.size(0) / batch_time_m.avg,
                        data_time=data_time_m)
                )

            if ((step+1) % eval_interval == 0 and step != 0) or (step+1) == num_training_steps:
                eval_metrics, eval_thresholds = evaluate_and_compute_threshold(
                    model=model,
                    dataloader=validloader,
                    criterion=criterion,
                    log_interval=log_interval,
                    metrics=[image_threshold, pixel_threshold],
                    top_k=top_k,
                    compute_threshold=compute_threshold,
                    device=device
                )
                model.train()
                eval_log = dict([(f'eval_{k}', v)
                                for k, v in eval_metrics.items()])

                # wandb
                if use_wandb:
                    wandb.log(eval_log, step=step)

                # checkpoint
                if best_score <= np.mean(list(eval_metrics.values())):
                    _logger.info('Best Score {0:.3%} to {1:.3%}'.format(
                        best_score, np.mean(list(eval_metrics.values()))))

                    best_score = np.mean(list(eval_metrics.values()))

                    # save best score
                    state = {'best_step': step}
                    state.update({'best_score': best_score})
                    state.update(eval_log)

                    # threshold
                    if compute_threshold:
                        eval_log = dict(
                            [(f'eval_{k}', v) for k, v in eval_thresholds.items()])
                        state.update(eval_log)

                        # wandb
                        if use_wandb:
                            wandb.log(eval_log, step=step)

                    json.dump(state, open(os.path.join(
                        savedir, 'best_score.json'), 'w'), indent='\t')

                    #  save best model
                    save_model(
                        path=os.path.join(savedir, save_model_path),
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        best_score=best_score
                    )

            # scheduler
            if scheduler:
                scheduler.step()

            end = time.time()

            step += 1

            if step == num_training_steps:
                train_mode = False
                break

    # print best score and step
    _logger.info('Best Metric: {0:.3%} (step {1:})'.format(
        best_score, state['best_step']))


def evaluate_and_compute_threshold(model, dataloader, criterion, log_interval, metrics: list, top_k: int = 100, compute_threshold: bool = False, device: str = 'cpu'):
    # metrics
    image_threshold, pixel_threshold = metrics

    # reset
    image_threshold.reset()
    pixel_threshold.reset()

    # targets and outputs
    image_targets = []
    image_masks = []
    anomaly_score = []
    anomaly_map = []

    model.eval()

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(tqdm(dataloader)):
            inputs, masks, labels = inputs.to(device, dtype=torch.float32), targets['mask'].to(
                device), targets['label'].to(device)

            # predict
            y_preds = model(inputs)
            anomaly_score_i = torch.topk(torch.flatten(
                y_preds, start_dim=1), top_k)[0].mean(dim=1)

            if compute_threshold:
                image_threshold.update(anomaly_score_i, labels)
                pixel_threshold.update(y_preds, masks)

            # stack targets and outputs
            image_targets.extend(labels.cpu().tolist())
            image_masks.extend(masks.cpu().numpy())

            anomaly_score.extend(anomaly_score_i.cpu().tolist())
            anomaly_map.extend(y_preds.cpu().numpy())

    # metrics
    image_masks = np.array(image_masks)
    anomaly_map = np.array(anomaly_map)
    auroc_image = roc_auc_score(image_targets, anomaly_score)
    auroc_pixel = roc_auc_score(
        image_masks.reshape(-1), anomaly_map.reshape(-1))
    all_fprs, all_pros = compute_pro(
        anomaly_maps=anomaly_map,
        ground_truth_maps=image_masks
    )

    aupro = trapezoid(all_fprs, all_pros)

    metrics = {
        'AUROC-image': auroc_image,
        'AUROC-pixel': auroc_pixel,
        'AUPRO-pixel': aupro

    }

    _logger.info('TEST: AUROC-image: {0:.3%} | AUROC-pixel: {1:.3%} | AUPRO-pixel: {2:.3%}'.format
                 (metrics['AUROC-image'], metrics['AUROC-pixel'], metrics['AUPRO-pixel']))

    eval_thresholds = {}
    if compute_threshold:
        eval_thresholds = {
            'Threshold-image': image_threshold.compute().item(),
            'Threshold-pixel': pixel_threshold.compute().item()
        }
        _logger.info('THRESHOLD: Threshold-image: %.3f | Threshold-pixel: %.3f' %
                     (eval_thresholds['Threshold-image'], eval_thresholds['Threshold-pixel']))

    return metrics, eval_thresholds
