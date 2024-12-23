import os
import onnxruntime
from tqdm import tqdm
import matplotlib.pyplot as plt
from ..data import create_dataset


def load_model_onnx(cfg, model_name, target):
    testset = create_dataset(
        datadir=cfg.DATASET.datadir,
        target=target,
        is_train=False,
        num_train=cfg.TRAIN.num_train,
        threshold=cfg.DATASET.threshold,
        resize=cfg.DATASET.resize,
    )
    cfg.EXP_NAME = cfg.EXP_NAME + f"-{model_name.split('-')[1]}"
    savedir = os.path.join(cfg.RESULT.savedir, cfg.TASK,
                           cfg.METHOD, target, cfg.EXP_NAME)
    resume = cfg.RESUME.onnxmodel

    # build Edge model
    use_gpu = cfg.TRAIN.use_gpu
    providers = ['CUDAExecutionProvider'] if use_gpu else [
        'CPUExecutionProvider']
    onnx_model_path = os.path.join(savedir, resume)
    model = onnxruntime.InferenceSession(onnx_model_path, providers=providers)

    params = {
        "savedir": savedir
    }

    return testset, model, params


def result_plot_onnx(idx, model, testset, params, test_img):
    if test_img:
        input_i = testset[idx]
    else:
        input_i, target_i, _ = testset[idx]

    # run the ONNX model
    output_i = model.run(None, {'input': input_i.unsqueeze(0).numpy()})

    output_i = output_i[-1].squeeze()

    fig, ax = plt.subplots(1, 2 if test_img else 3, figsize=(10, 10))

    # show image
    ax[0].imshow(input_i.permute(1, 2, 0))
    ax[0].set_title('Image')

    # show predict
    ax[1].imshow(output_i, cmap='gray')
    ax[1].set_title('Edge Predicted Mask')

    if not test_img:
        # show gt truth
        ax[2].imshow(target_i.squeeze(), cmap='gray')
        ax[2].set_title('Edge Truth Mask')

    plt.show()


def result_save_plot_onnx(cfg, model, testset, params, test_img):
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
            input_i, target_i, _ = testset[idx]

        # run the ONNX model
        output_i = model.run(None, {'input': input_i.unsqueeze(0).numpy()})

        output_i = output_i[-1].squeeze()

        fig, ax = plt.subplots(1, 2 if test_img else 3, figsize=(10, 10))

        # show image
        ax[0].imshow(input_i.permute(1, 2, 0))
        ax[0].set_title('Image')

        # show predict
        ax[1].imshow(output_i, cmap='gray')
        ax[1].set_title('Edge Predicted Mask')

        if not test_img:
            # show gt truth
            ax[2].imshow(target_i.squeeze(), cmap='gray')
            ax[2].set_title('Edge Truth Mask')

        plt.savefig(save_path)
        plt.close(fig)
