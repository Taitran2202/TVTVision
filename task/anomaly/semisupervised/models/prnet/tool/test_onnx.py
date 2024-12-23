import os
import json
import onnxruntime
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import torch
from utils.post_processing.anomaly.anomaly_process import *
from ....data import create_dataset


def load_model_onnx(cfg, model_name, target):
    testset = create_dataset(
        datadir=cfg.DATASET.datadir,
        target=target,
        is_train=False,
        resize=cfg.DATASET.resize
    )
    cfg.EXP_NAME = cfg.EXP_NAME + f"-{model_name.split('-')[1]}"
    savedir = os.path.join(cfg.RESULT.savedir, cfg.TASK,
                           cfg.METHOD, target, cfg.EXP_NAME)
    resume = cfg.RESUME.onnxmodel

    # build PRNet
    use_gpu = cfg.TRAIN.use_gpu
    providers = ['CUDAExecutionProvider'] if use_gpu else [
        'CPUExecutionProvider']
    onnx_model_path = os.path.join(savedir, resume)
    model = onnxruntime.InferenceSession(onnx_model_path, providers=providers)

    # load threshold
    if cfg.TRAIN.compute_threshold:
        # opening json file
        f = open(os.path.join(savedir, 'best_score.json'))

        # returns JSON object as
        # a dictionary
        data = json.load(f)

        threshold_image = data['eval_Threshold-image']
        threshold_pixel = data['eval_Threshold-pixel']

    else:
        threshold_image = cfg.TEST.threshold_image
        threshold_pixel = cfg.TEST.threshold_pixel

    params = {
        "threshold_image": threshold_image,
        "threshold_pixel": threshold_pixel,
        "savedir": savedir
    }

    return testset, model, params


def result_plot_onnx(idx, model, testset, params, test_img):
    threshold_image = params['threshold_image']
    threshold_pixel = params['threshold_pixel']

    if test_img:
        input_i = testset[idx]
    else:
        input_i, target_i = testset[idx]

    input_i = input_i.unsqueeze(0).numpy()  # convert to numpy array
    output_i = model.run(None, {'input': input_i})[0]  # run the ONNX model

    # convert output back to torch tensor
    output_i = torch.from_numpy(output_i).squeeze()
    heat_map = compute_anomaly_mask(output_i, threshold_pixel)

    prediction = compute_anomaly_predict(output_i, threshold_image)

    fig, ax = plt.subplots(1, 4 if test_img else 5, figsize=(15, 10))

    # show image
    ax[0].imshow(input_i.squeeze().transpose(1, 2, 0))
    ax[0].set_title('Input:' if test_img else 'Input: {}'.format(
        'Normal' if target_i['label'] == 0 else 'Abnormal'))

    # show gt predicted
    ax[1].imshow(output_i, cmap='gray')
    ax[1].set_title('Predicted Mask')

    # apply mask for image
    ax[2].imshow(input_i.squeeze().transpose(1, 2, 0), alpha=1)
    ax[2].imshow(heat_map, cmap='gray', alpha=0.5)
    ax[2].set_title(f'Anomaly Detected: {prediction}')

    # segmentation result
    segmentations = mark_boundaries(
        input_i.squeeze().transpose(1, 2, 0), heat_map, color=(1, 0, 0), mode="thick")
    ax[3].imshow(segmentations)
    ax[3].set_title('Segmentation Result')

    if not test_img:
        # show gt truth
        ax[4].imshow(target_i['mask'], cmap='gray')
        ax[4].set_title('Ground Truth')

    plt.show()


def result_save_plot_onnx(cfg, model, testset, params, test_img):
    threshold_image = params['threshold_image']
    threshold_pixel = params['threshold_pixel']

    if not test_img:
        savedir = os.path.join(params['savedir'], cfg.RESULT.saveallonnx)
    else:
        savedir = os.path.join(
            params['savedir'], cfg.RESULT.save_test_img_onnx)
    os.makedirs(savedir, exist_ok=True)

    for idx, file_path in enumerate(tqdm(testset.file_list)):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        file_prefix = os.path.basename(os.path.dirname(file_path))
        save_path = f"{savedir}/result_{file_prefix}_{file_name}.png"

        if test_img:
            input_i = testset[idx]
        else:
            input_i, target_i = testset[idx]

        input_i = input_i.unsqueeze(0).numpy()  # convert to numpy array
        output_i = model.run(None, {'input': input_i})[0]  # run the ONNX model

        # convert output back to torch tensor
        output_i = torch.from_numpy(output_i).squeeze()
        heat_map = compute_anomaly_mask(output_i, threshold_pixel)

        prediction = compute_anomaly_predict(output_i, threshold_image)

        fig, ax = plt.subplots(1, 4 if test_img else 5, figsize=(15, 10))

        # show image
        ax[0].imshow(input_i.squeeze().transpose(1, 2, 0))
        ax[0].set_title('Input:' if test_img else 'Input: {}'.format(
            'Normal' if target_i['label'] == 0 else 'Abnormal'))

        # show gt predicted
        ax[1].imshow(output_i, cmap='gray')
        ax[1].set_title('Predicted Mask')

        # apply mask for image
        ax[2].imshow(input_i.squeeze().transpose(1, 2, 0), alpha=1)
        ax[2].imshow(heat_map, cmap='gray', alpha=0.5)
        ax[2].set_title(f'Anomaly Detected: {prediction}')

        # segmentation result
        segmentations = mark_boundaries(
            input_i.squeeze().transpose(1, 2, 0), heat_map, color=(1, 0, 0), mode="thick")
        ax[3].imshow(segmentations)
        ax[3].set_title('Segmentation Result')

        if not test_img:
            # show gt truth
            ax[4].imshow(target_i['mask'], cmap='gray')
            ax[4].set_title('Ground Truth')

        plt.savefig(save_path)
        plt.close(fig)
