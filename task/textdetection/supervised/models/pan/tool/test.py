import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from ..model import PAN
from ..data import create_synth_dataset, create_icdar_dataset
from utils.post_processing.textdetection.pan import test_net


def load_model(cfg, model_name, target):
    if cfg.DATASET.test_folder == 'synth':
        testset = create_synth_dataset(
            datadir=cfg.DATASET.datadir,
            target=cfg.DATASET.test_folder,
            is_train=False,
            aug=cfg.DATASET.syn_aug,
            resize=cfg.DATASET.resize,
            short_size=cfg.DATASET.short_size,
            kernel_scale=cfg.DATASET.kernel_scale
        )
    elif 'icdar' in cfg.DATASET.test_folder:
        testset = create_icdar_dataset(
            datadir=cfg.DATASET.datadir,
            target=cfg.DATASET.test_folder,
            is_train=False,
            aug=cfg.DATASET.icdar_aug,
            resize=cfg.DATASET.resize,
            short_size=cfg.DATASET.short_size,
            kernel_scale=cfg.DATASET.kernel_scale
        )
    global device
    use_gpu = cfg.TRAIN.use_gpu
    device = torch.device(
        "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    # build PAN
    model = PAN(
        pretrained=cfg['MODEL']['pretrained'],
        neck_channel=cfg['MODEL']['neck_channel'],
        pa_in_channels=cfg['MODEL']['pa_in_channels'],
        hidden_dim=cfg['MODEL']['hidden_dim'],
        num_classes=cfg['MODEL']['num_classes']
    )
    model.to(device)

    cfg.EXP_NAME = cfg.EXP_NAME + f"-{model_name.split('-')[1]}"
    savedir = os.path.join(cfg.RESULT.savedir, cfg.TASK,
                           cfg.METHOD, target, cfg.EXP_NAME)
    resume = cfg.RESUME.bestmodel
    checkpoint = torch.load(f'{savedir}/{resume}', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    params = {
        'min_score': cfg.TEST.min_score,
        'min_area': cfg.TEST.min_area,
        'bbox_type': cfg.TEST.bbox_type,
        'resize': cfg.DATASET.resize,
        'target': cfg.DATASET.test_folder,
        'savedir': savedir
    }
    return testset, model, params


def result_plot(idx, model, testset, params, test_img):
    if test_img:
        img_path = testset.file_list[idx]
    else:
        img_path, single_img_bboxes = testset[idx]
    input_i = cv2.imread(img_path)
    input_i = cv2.cvtColor(input_i, cv2.COLOR_BGR2RGB)

    bboxes, _ = test_net(
        model=model,
        image=input_i,
        device=device,
        min_area=params['min_area'],
        min_score=params['min_score'],
        bbox_type=params['bbox_type'],
        resize=params['resize'],
    )

    if not test_img:
        img = input_i.copy()
        if params['target'] == 'icdar13':
            # draw bounding boxes for gt, color red
            if single_img_bboxes is not None:
                for j in range(len(single_img_bboxes)):
                    cv2.polylines(
                        img,
                        [np.array(single_img_bboxes[j]["points"]
                                  ).reshape((-1, 1, 2))],
                        True,
                        color=(0, 0, 255),
                        thickness=2,
                    )
        elif params['target'] == 'icdar15':
            # draw bounding boxes for gt, color red
            if single_img_bboxes is not None:
                for j in range(len(single_img_bboxes)):
                    _gt_box = np.array(
                        single_img_bboxes[j]["points"]).reshape(-1, 2).astype(np.int32)
                    if single_img_bboxes[j]["text"] == "###":
                        cv2.polylines(img, [_gt_box], True, color=(
                            128, 128, 128), thickness=2)
                    else:
                        cv2.polylines(img, [_gt_box], True,
                                      color=(0, 0, 255), thickness=2)
        elif params['target'] == 'icdar17':
            # draw bounding boxes for gt, color red
            if single_img_bboxes is not None:
                for j in range(len(single_img_bboxes)):
                    _gt_box = np.array(
                        single_img_bboxes[j]["points"]).reshape(-1, 2).astype(np.int32)
                    if single_img_bboxes[j]["text"] == "###":
                        cv2.polylines(img, [_gt_box], True, color=(
                            128, 128, 128), thickness=2)
                    else:
                        cv2.polylines(img, [_gt_box], True,
                                      color=(0, 0, 255), thickness=2)

    img1 = input_i.copy()
    for _, box in enumerate(bboxes):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        cv2.polylines(img1, [poly.reshape((-1, 1, 2))],
                      True, color=(0, 255, 0), thickness=2)

    fig, ax = plt.subplots(1, 2 if test_img else 3, figsize=(15, 10))
    ax[0].imshow(input_i)
    ax[0].set_title('Input')
    ax[1].imshow(img1)
    ax[1].set_title('Box Predicted')
    if not test_img:
        ax[2].imshow(img)
        ax[2].set_title('Box Truth')

    plt.show()


def result_save_plot(cfg, model, testset, params, test_img):
    if not test_img:
        savedir = os.path.join(params['savedir'], cfg.RESULT.saveall)
    else:
        savedir = os.path.join(params['savedir'], cfg.RESULT.save_test_img)
    os.makedirs(savedir, exist_ok=True)

    for idx, file_path in enumerate(tqdm(testset.file_list)):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        file_prefix = os.path.basename(os.path.dirname(file_path))
        save_path = f"{savedir}/result_{file_prefix}_{file_name}.png"

        if test_img:
            img_path = file_path
        else:
            img_path, single_img_bboxes = testset[idx]
        input_i = cv2.imread(img_path)
        input_i = cv2.cvtColor(input_i, cv2.COLOR_BGR2RGB)

        bboxes, _ = test_net(
            model=model,
            image=input_i,
            device=device,
            min_area=params['min_area'],
            min_score=params['min_score'],
            bbox_type=params['bbox_type'],
            resize=params['resize'],
        )

        if not test_img:
            img = input_i.copy()
            if params['target'] == 'icdar13':
                # draw bounding boxes for gt, color red
                if single_img_bboxes is not None:
                    for j in range(len(single_img_bboxes)):
                        cv2.polylines(
                            img,
                            [np.array(single_img_bboxes[j]["points"]
                                      ).reshape((-1, 1, 2))],
                            True,
                            color=(0, 0, 255),
                            thickness=2,
                        )
            elif params['target'] == 'icdar15':
                # draw bounding boxes for gt, color red
                if single_img_bboxes is not None:
                    for j in range(len(single_img_bboxes)):
                        _gt_box = np.array(
                            single_img_bboxes[j]["points"]).reshape(-1, 2).astype(np.int32)
                        if single_img_bboxes[j]["text"] == "###":
                            cv2.polylines(img, [_gt_box], True, color=(
                                128, 128, 128), thickness=2)
                        else:
                            cv2.polylines(img, [_gt_box], True,
                                          color=(0, 0, 255), thickness=2)
            elif params['target'] == 'icdar17':
                # draw bounding boxes for gt, color red
                if single_img_bboxes is not None:
                    for j in range(len(single_img_bboxes)):
                        _gt_box = np.array(
                            single_img_bboxes[j]["points"]).reshape(-1, 2).astype(np.int32)
                        if single_img_bboxes[j]["text"] == "###":
                            cv2.polylines(img, [_gt_box], True, color=(
                                128, 128, 128), thickness=2)
                        else:
                            cv2.polylines(img, [_gt_box], True,
                                          color=(0, 0, 255), thickness=2)

        img1 = input_i.copy()
        for _, box in enumerate(bboxes):
            poly = np.array(box).astype(np.int32).reshape((-1))
            poly = poly.reshape(-1, 2)
            cv2.polylines(img1, [poly.reshape((-1, 1, 2))],
                          True, color=(0, 255, 0), thickness=2)

        fig, ax = plt.subplots(1, 2 if test_img else 3, figsize=(15, 10))
        ax[0].imshow(input_i)
        ax[0].set_title('Input')
        ax[1].imshow(img1)
        ax[1].set_title('Box Predicted')
        if not test_img:
            ax[2].imshow(img)
            ax[2].set_title('Box Truth')

        plt.savefig(save_path)
        plt.close(fig)