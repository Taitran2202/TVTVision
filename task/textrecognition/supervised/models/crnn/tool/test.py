import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from ..model import CRNN
from ..data import create_synth90k_dataset, create_icdar_dataset
from utils.post_processing.textrecognition.crnn import ctc_decode


def load_model(cfg, model_name, target):
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
    global device
    use_gpu = cfg['TRAIN']['use_gpu']
    # device = torch.device(
    #     "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # build CRNN
    model = CRNN(
        img_channel=cfg['MODEL']['img_channel'],
        img_height=cfg['DATASET']['resize'][0],
        img_width=cfg['DATASET']['resize'][1],
        num_class=len(cfg['DATASET']['chars']) + 1,
        map_to_seq_hidden=cfg['MODEL']['map_to_seq_hidden'],
        rnn_hidden=cfg['MODEL']['rnn_hidden'],
        leaky_relu=cfg['MODEL']['leaky_relu'],
    )
    model.to(device)

    cfg['EXP_NAME'] = cfg['EXP_NAME'] + f"-{model_name.split('-')[1]}"
    savedir = os.path.join(
        cfg['RESULT']['savedir'], cfg['TASK'], cfg['METHOD'], cfg['DATASET']['target'], cfg['EXP_NAME'])
    resume = cfg['RESUME']['bestmodel']
    checkpoint = torch.load(f'{savedir}/{resume}', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    params = {
        'method': cfg['TRAIN']['decode_method'],
        'beam_size': cfg['TRAIN']['beam_size'],
        'label2char': testset.label2char,
        'savedir': savedir
    }
    return testset, model, params


def result_plot(idx, model, testset, params, test_img):
    method = params['method']
    beam_size = params['beam_size']
    label2char = params['label2char']

    if test_img:
        input_i = testset[idx]
    else:
        input_i, targets_i, _ = testset[idx]
        text = [label2char[c] for c in targets_i.cpu().numpy()]

    output_i = model(input_i.unsqueeze(0).to(
        device, dtype=torch.float32)).cpu().detach().numpy()
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


def result_save_plot(cfg, model, testset, params, test_img):
    method = params['method']
    beam_size = params['beam_size']
    label2char = params['label2char']

    if not test_img:
        savedir = os.path.join(params['savedir'], cfg['RESULT']['saveall'])
    else:
        savedir = os.path.join(
            params['savedir'], cfg['RESULT']['save_test_img'])
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

        output_i = model(input_i.unsqueeze(0).to(
            device, dtype=torch.float32)).cpu().detach().numpy()
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
