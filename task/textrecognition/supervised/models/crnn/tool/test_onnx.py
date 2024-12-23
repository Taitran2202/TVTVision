import os
import onnxruntime
from tqdm import tqdm
import matplotlib.pyplot as plt
from ..data import create_synth90k_dataset, create_icdar_dataset
from utils.post_processing.textrecognition.crnn import ctc_decode


def load_model_onnx(cfg, model_name, target):
    if target == 'synth90k':
        testset = create_synth90k_dataset(
            datadir=cfg['DATASET']['datadir'],
            target=target,
            train=False,
            chars=cfg['DATASET']['chars'],
            resize=cfg['DATASET']['resize'],
        )
    else:
        testset = create_icdar_dataset(
            datadir=cfg['DATASET']['datadir'],
            target=target,
            train=False,
            chars=cfg['DATASET']['chars'],
            resize=cfg['DATASET']['resize'],
        )
    cfg['EXP_NAME'] = cfg['EXP_NAME'] + f"-{model_name.split('-')[1]}"
    savedir = os.path.join(
        cfg['RESULT']['savedir'], cfg['TASK'], cfg['METHOD'], cfg['DATASET']['target'], cfg['EXP_NAME'])
    resume = cfg['RESUME']['onnxmodel']

    # build CRNN
    use_gpu = cfg['TRAIN']['use_gpu']
    providers = ['CUDAExecutionProvider'] if use_gpu else [
        'CPUExecutionProvider']
    onnx_model_path = os.path.join(savedir, resume)
    model = onnxruntime.InferenceSession(onnx_model_path, providers=providers)

    params = {
        'method': cfg['TRAIN']['decode_method'],
        'beam_size': cfg['TRAIN']['beam_size'],
        'label2char': testset.label2char,
        'savedir': savedir
    }

    return testset, model, params


def result_plot_onnx(idx, model, testset, params, test_img):
    method = params['method']
    beam_size = params['beam_size']
    label2char = params['label2char']

    if test_img:
        input_i = testset[idx]
    else:
        input_i, targets_i, _ = testset[idx]
        text = [label2char[c] for c in targets_i.cpu().numpy()]

    # run the ONNX model
    output_i = model.run(None, {'input': input_i.unsqueeze(0).numpy()})[0]
    text_pred = ctc_decode(output_i, label2char=label2char,
                           method=method, beam_size=beam_size)

    fig, ax = plt.subplots(1, 2, figsize=(15, 10))

    # show image
    ax[0].imshow(input_i.permute(1, 2, 0), cmap='gray')
    ax[0].set_title('Input:' if test_img else 'Text: {}'.format(text))

    # show predicted
    ax[1].imshow(input_i.permute(1, 2, 0), cmap='gray')
    ax[1].set_title('Predicted Text: {}'.format(text_pred))

    plt.show()


def result_save_plot_onnx(cfg, model, testset, params, test_img):
    method = params['method']
    beam_size = params['beam_size']
    label2char = params['label2char']

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
            input_i = testset[idx]
        else:
            input_i, targets_i, _ = testset[idx]
            text = [label2char[c] for c in targets_i.cpu().numpy()]

        # run the ONNX model
        output_i = model.run(None, {'input': input_i.unsqueeze(0).numpy()})[0]
        text_pred = ctc_decode(output_i, label2char=label2char,
                               method=method, beam_size=beam_size)

        fig, ax = plt.subplots(1, 2, figsize=(15, 10))

        # show image
        ax[0].imshow(input_i.permute(1, 2, 0), cmap='gray')
        ax[0].set_title('Input:' if test_img else 'Text: {}'.format(text))

        # show predicted
        ax[1].imshow(input_i.permute(1, 2, 0), cmap='gray')
        ax[1].set_title('Predicted Text: {}'.format(text_pred))

        plt.savefig(save_path)
        plt.close(fig)
