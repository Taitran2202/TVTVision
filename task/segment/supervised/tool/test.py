import os
import importlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import torch
from ..data import create_dataset
from utils.post_processing.segment.segment_process import compute_segment_mask


def load_model(cfg, model_name, target):
    testset = create_dataset(
        datadir=cfg['DATASET']['datadir'],
        target=target,
        is_train=False,
        aug=cfg['DATASET']['aug'],
        num_train=cfg['TRAIN']['num_train'],
        resize=cfg['DATASET']['resize'],
    )
    global device
    use_gpu = cfg['TRAIN']['use_gpu']
    device = torch.device(
        "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    # build Model
    model_path = f"task.segment.supervised.models.{cfg['EXP_NAME'].split('-')[0].lower()}"
    model = importlib.import_module(model_path).build_model(cfg=cfg)
    model.to(device)

    cfg['EXP_NAME'] = cfg['EXP_NAME'] + f"-{model_name.split('-')[1]}"
    savedir = os.path.join(
        cfg['RESULT']['savedir'], cfg['TASK'], cfg['METHOD'], cfg['DATASET']['target'], cfg['EXP_NAME'])
    resume = cfg['RESUME']['bestmodel']
    checkpoint = torch.load(f'{savedir}/{resume}', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return testset, model, None


def result_plot(idx, model, testset, params, test_img):
    if test_img:
        input_i = testset[idx]
    else:
        input_i, targets_i = testset[idx]

    output_i = model(input_i.unsqueeze(0).to(
        device, dtype=torch.float32)).cpu().detach()[0]

    heat_map = compute_segment_mask(output_i, 0.5)

    fig, ax = plt.subplots(1, 3 if test_img else 4, figsize=(15, 10))

    # show image
    ax[0].imshow(input_i.permute(1, 2, 0))
    ax[0].set_title('Input')

    # show gt predicted
    ax[1].imshow(output_i[0], cmap='gray')
    ax[1].set_title('Predicted Mask')

    # segmentation result
    segmentations = mark_boundaries(
        input_i.permute(1, 2, 0).numpy(), heat_map, color=(1, 0, 0), mode="thick")
    ax[2].imshow(segmentations)
    ax[2].set_title('Segmentation Result')

    if not test_img:
        # show gt truth
        ax[3].imshow(targets_i, cmap='gray')
        ax[3].set_title('Ground Truth')

    plt.show()


def result_save_plot(cfg, model, testset, params, test_img):
    if not test_img:
        savedir = os.path.join(cfg['RESULT']['savedir'], cfg['TASK'], cfg['METHOD'],
                               cfg['DATASET']['target'], cfg['EXP_NAME'], cfg['RESULT']['saveall'])
    else:
        savedir = os.path.join(cfg['RESULT']['savedir'], cfg['TASK'], cfg['METHOD'],
                               cfg['DATASET']['target'], cfg['EXP_NAME'], cfg['RESULT']['save_test_img'])
    os.makedirs(savedir, exist_ok=True)

    for idx, file_path in enumerate(tqdm(testset.file_list)):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        file_prefix = os.path.basename(os.path.dirname(file_path))
        save_path = f"{savedir}/result_{file_prefix}_{file_name}.png"

        if test_img:
            input_i = testset[idx]
        else:
            input_i, targets_i = testset[idx]

        output_i = model(input_i.unsqueeze(0).to(
            device, dtype=torch.float32)).cpu().detach()[0]

        heat_map = compute_segment_mask(output_i, 0.5)

        fig, ax = plt.subplots(1, 3 if test_img else 4, figsize=(15, 10))

        # show image
        ax[0].imshow(input_i.permute(1, 2, 0))
        ax[0].set_title('Input')

        # show gt predicted
        ax[1].imshow(output_i[0], cmap='gray')
        ax[1].set_title('Predicted Mask')

        # segmentation result
        segmentations = mark_boundaries(
            input_i.permute(1, 2, 0).numpy(), heat_map, color=(1, 0, 0), mode="thick")
        ax[2].imshow(segmentations)
        ax[2].set_title('Segmentation Result')

        if not test_img:
            # show gt truth
            ax[3].imshow(targets_i, cmap='gray')
            ax[3].set_title('Ground Truth')

        plt.savefig(save_path)
        plt.close(fig)
