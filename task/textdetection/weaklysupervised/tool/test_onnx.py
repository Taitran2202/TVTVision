import os
import cv2
import numpy as np
import onnxruntime
from tqdm import tqdm
import matplotlib.pyplot as plt
from ..data import create_icdar_dataset, create_synth_dataset
from utils.post_processing.textdetection.weaklysupervised import test_onnx_net


def load_model_onnx(cfg, model_name, target):
    if cfg['DATASET']['test_folder'] == 'synth':
        testset = create_synth_dataset(
            datadir=cfg['DATASET']['datadir'],
            target=cfg['DATASET']['test_folder'],
            is_train=False,
            aug=cfg['DATASET']['syn_aug'],
            resize=cfg['DATASET']['resize'],
            gauss_init_size=cfg['DATASET']['gauss_init_size'],
            gauss_sigma=cfg['DATASET']['gauss_sigma'],
            enlarge_region=cfg['DATASET']['enlarge_region'],
            enlarge_affinity=cfg['DATASET']['enlarge_affinity'],
        )
    else:
        testset = create_icdar_dataset(
            datadir=cfg['DATASET']['datadir'],
            target=cfg['DATASET']['test_folder'],
            is_train=False,
            aug=cfg['DATASET']['icdar_aug'],
            vis_train_dir=None,
            pseudo_vis_opt=False,
            net=None,
            do_not_care_label=cfg['DATASET']['do_not_care_label'],
            device=None,
            resize=cfg['DATASET']['resize'],
            gauss_init_size=cfg['DATASET']['gauss_init_size'],
            gauss_sigma=cfg['DATASET']['gauss_sigma'],
            enlarge_region=cfg['DATASET']['enlarge_region'],
            enlarge_affinity=cfg['DATASET']['enlarge_affinity'],
        )
    cfg['EXP_NAME'] = cfg['EXP_NAME'] + f"-{model_name.split('-')[1]}"
    savedir = os.path.join(
        cfg['RESULT']['savedir'], cfg['TASK'], cfg['METHOD'], target, cfg['EXP_NAME'])
    resume = cfg['RESUME']['onnxmodel']

    # build Model
    use_gpu = cfg['TRAIN']['use_gpu']
    providers = ['CUDAExecutionProvider'] if use_gpu else [
        'CPUExecutionProvider']
    onnx_model_path = os.path.join(savedir, resume)
    model = onnxruntime.InferenceSession(onnx_model_path, providers=providers)

    params = {
        'text_threshold': cfg['TEST']['text_threshold'],
        'link_threshold': cfg['TEST']['link_threshold'],
        'low_text': cfg['TEST']['low_text'],
        'resize': cfg['DATASET']['resize'],
        'target': cfg['DATASET']['test_folder'],
        'savedir': savedir
    }

    return testset, model, params


def result_plot_onnx(idx, model, testset, params, test_img):
    if test_img:
        img_path = testset.file_list[idx]
    else:
        img_path, single_img_bboxes = testset[idx]
    input_i = cv2.imread(img_path)
    input_i = cv2.cvtColor(input_i, cv2.COLOR_BGR2RGB)

    bboxes, _ = test_onnx_net(
        model,
        input_i,
        params['text_threshold'],
        params['link_threshold'],
        params['low_text'],
        params['resize']
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


def result_save_plot_onnx(cfg, model, testset, params, test_img):
    if not test_img:
        savedir = os.path.join(params['savedir'], cfg['RESULT']['saveallonnx'])
    else:
        savedir = os.path.join(
            params['savedir'], cfg['RESULT']['save_test_img_onnx'])
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

        bboxes, _ = test_onnx_net(
            model,
            input_i,
            params['text_threshold'],
            params['link_threshold'],
            params['low_text'],
            params['resize'],
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
