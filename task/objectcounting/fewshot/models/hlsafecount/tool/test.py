import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from ..model import HLSAFECount
from ....data import create_hlsafecount_dataset


def load_model(cfg, model_name, target):
    trainset = create_hlsafecount_dataset(
        datadir=cfg.DATASET.datadir,
        target=target,
        is_train=True,
        aug=cfg.DATASET.aug,
        num_boxes=cfg.DATASET.num_boxes,
        num_train=cfg.TRAIN.num_train,
        resize=cfg.DATASET.resize,
    )
    image_support, _, boxes_support, _ = trainset[0]
    testset = create_hlsafecount_dataset(
        datadir=cfg.DATASET.datadir,
        target=target,
        is_train=False,
        aug=cfg.DATASET.aug,
        num_boxes=cfg.DATASET.num_boxes,
        num_train=cfg.TRAIN.num_train,
        resize=cfg.DATASET.resize,
    )
    global device
    use_gpu = cfg.TRAIN.use_gpu
    device = torch.device(
        "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    # build HLSAFECount
    model = HLSAFECount(
        num_block=cfg.MODEL.block,
        backbone_type=cfg.MODEL.backbone,
        backbone_out_layers=cfg.MODEL.backbone_out_layers,
        backbone_out_stride=cfg.MODEL.backbone_out_stride,
        pretrained=cfg.MODEL.pretrained,
        embed_dim=cfg.MODEL.embed_dim,
        mid_dim=cfg.MODEL.mid_dim,
        head=cfg.MODEL.head,
        dropout=cfg.MODEL.dropout,
        exemplar_scales=cfg.MODEL.exemplar_scales,
        image_support=image_support.unsqueeze(0),
        boxes_support=boxes_support.unsqueeze(0)
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
        "savedir": savedir,
    }
    return testset, model, params


def result_plot(idx, model, testset, params, test_img):
    if test_img:
        input_i = testset[idx]
    else:
        input_i, _, _, cnt_i = testset[idx]

    density_pred, cnt_pred = model(input_i.unsqueeze(0).to(device))
    density_pred = density_pred.cpu().detach().numpy()
    cnt_pred = cnt_pred.cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(15, 10))

    # show truth
    ax[0].imshow(input_i.permute(1, 2, 0))
    ax[0].set_title(
        'Input: ' if test_img else 'Count Truth: {}'.format(cnt_i.cpu().numpy()))

    # show predicted
    ax[1].imshow(density_pred[0][0])
    ax[1].set_title('Count Predict: {}'.format(cnt_pred))

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
            input_i = testset[idx]
        else:
            input_i, _, _, cnt_i = testset[idx]

        density_pred, cnt_pred = model(input_i.unsqueeze(0).to(device))
        density_pred = density_pred.cpu().detach().numpy()
        cnt_pred = cnt_pred.cpu().numpy()

        fig, ax = plt.subplots(1, 2, figsize=(15, 10))

        # show truth
        ax[0].imshow(input_i.permute(1, 2, 0))
        ax[0].set_title(
            'Input: ' if test_img else 'Count Truth: {}'.format(cnt_i.cpu().numpy()))

        # show predicted
        ax[1].imshow(density_pred[0][0])
        ax[1].set_title('Count Predict: {}'.format(cnt_pred))

        plt.savefig(save_path)
        plt.close(fig)
