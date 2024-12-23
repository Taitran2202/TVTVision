import os
import json
import importlib
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from ..data import create_dataset


def load_model(cfg, model_name, target):
    testset = create_dataset(
        datadir=cfg.DATASET.datadir,
        target=target,
        is_train=False,
        aug=cfg.DATASET.aug,
        num_train=cfg.TRAIN.num_train,
        resize=cfg.DATASET.resize,
    )
    global device
    use_gpu = cfg.TRAIN.use_gpu
    device = torch.device(
        "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    # build Model
    model_path = f"task.classify.supervised.models.{cfg.EXP_NAME.split('-')[0].lower()}"
    model = importlib.import_module(model_path).build_model(
        cfg=cfg, cls=testset.classes)
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
        "classes": testset.classes,
        "savedir": savedir,
    }
    return testset, model, params


def result_plot(idx, model, testset, params, test_img):
    if test_img:
        input_i = testset[idx]
        classes = params['classes']
    else:
        input_i, label_i = testset[idx]

    output_i = model(input_i.unsqueeze(0).to(
        device, dtype=torch.float32)).cpu().detach()

    _, preds = torch.max(output_i, dim=1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    # show image
    ax[0].imshow(input_i.permute(1, 2, 0))
    ax[0].set_title('Input:' if test_img else 'Input: {}'.format(
        testset.classes[label_i]))

    # show gt truth
    ax[1].imshow(input_i.permute(1, 2, 0))
    ax[1].set_title(classes[preds[0].item()] if test_img else 'Predicted: {}'.format(
        testset.classes[preds[0].item()]))

    plt.show()


def result_save_plot(cfg, model, testset, params, test_img):
    if not test_img:
        savedir = os.path.join(params['savedir'], cfg.RESULT.saveall)
    else:
        savedir = os.path.join(params['savedir'], cfg.RESULT.save_test_img)
    os.makedirs(savedir, exist_ok=True)

    for idx, (file_path, _) in enumerate(tqdm(testset.file_list)):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        file_prefix = os.path.basename(os.path.dirname(file_path))
        save_path = f"{savedir}/result_{file_prefix}_{file_name}.png"

        if test_img:
            input_i = testset[idx]
            classes = params['classes']
        else:
            input_i, label_i = testset[idx]

        output_i = model(input_i.unsqueeze(0).to(
            device, dtype=torch.float32)).cpu().detach()

        _, preds = torch.max(output_i, dim=1)

        fig, ax = plt.subplots(1, 2, figsize=(10, 10))

        # show image
        ax[0].imshow(input_i.permute(1, 2, 0))
        ax[0].set_title('Input:' if test_img else 'Input: {}'.format(
            testset.classes[label_i]))

        # show gt truth
        ax[1].imshow(input_i.permute(1, 2, 0))
        ax[1].set_title(classes[preds[0].item()] if test_img else 'Predicted: {}'.format(
            testset.classes[preds[0].item()]))

        plt.savefig(save_path)
        plt.close(fig)


def show_train_result(params):
    p_ce = os.path.join(params['savedir'], 'results-ce-per_class.json')
    p_bcr = os.path.join(params['savedir'], 'results-bcr-per_class.json')

    class_names = params['classes']
    if os.path.exists(p_ce):
        heatmap_ce = json.load(open(p_ce, 'r'))['cm']
        heatmap = heatmap_ce
        titles = 'CrossEntropyLoss'
    if os.path.exists(p_bcr):
        heatmap_bcr = json.load(open(p_bcr, 'r'))['cm']
        heatmap = heatmap_bcr
        titles = 'BalancedSoftmax'

    fig, ax = plt.subplots(1, 1)
    sns.heatmap(heatmap, annot=True, fmt='d', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_ylabel('Ground truth')
    ax.set_xlabel('Prediction')
    ax.set_title(titles)

    plt.tight_layout()
    plt.savefig(os.path.join(params['savedir'],
                'confusion_matrix.jpg'), dpi=300)
    plt.show()
