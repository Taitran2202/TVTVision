import os
import wandb
import warnings
import importlib
import logging
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .plot_image import plot_image
from .data import create_dataset, create_dataloader
from .utils.misc import replace_module, SiLU
from utils.export_onnx import export_to_onnx
from utils.seed import seeding
from utils.log import setup_default_logging
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler
_logger = logging.getLogger('train')
warnings.filterwarnings("ignore")


def visualize_augmentations(dataset, cfg, normalize_box):
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for j, (image, target) in enumerate(loader):
        if j == cfg.TRAIN.num_sample:
            break
        plot_image(image[0].permute(1, 2, 0), target,
                   cfg.DATASET.class_names, normalize_box)


def run(cfg):
    # savedir
    if 'model_type' in cfg.MODEL:
        model_name = cfg.MODEL.model_type
    else:
        model_name = cfg.MODEL.backbone
    cfg.EXP_NAME = cfg.EXP_NAME + f"-{model_name}"
    savedir = os.path.join(
        cfg.RESULT.savedir, cfg.TASK, cfg.METHOD, cfg.DATASET.target, cfg.EXP_NAME)
    vis_test_dir = os.path.join(
        cfg.RESULT.savedir, cfg.TASK, cfg.METHOD, cfg.DATASET.target, cfg.EXP_NAME, cfg.RESULT.vis_test_dir)
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(vis_test_dir, exist_ok=True)

    setup_default_logging(log_path=os.path.join(savedir, 'log.txt'))

    # setting seed and device
    seeding(cfg.SEED)

    use_gpu = cfg.TRAIN.use_gpu
    device = torch.device(
        "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    _logger.info('Device: {}'.format(device))

    # wandb
    api_key = cfg.API_KEY
    if api_key:
        wandb.login(key=api_key)
    else:
        wandb.login()

    if cfg.TRAIN.use_wandb:
        wandb.init(name=cfg.DATASET.target + '_' + cfg.EXP_NAME, project=cfg.EXP_NAME.split('-')
                   [0], entity='visionteam', config=OmegaConf.to_container(cfg))

    # Hyperparameters
    _logger.info(f"Image Size: {cfg.DATASET.resize}")
    _logger.info(f"Batch Size: {cfg.DATALOADER.batch_size}")
    _logger.info(f"Epochs: {cfg.TRAIN.epochs}")

    # build datasets
    _logger.info('Setting up data...')
    trainset = create_dataset(
        datadir=cfg.DATASET.datadir,
        target=cfg.DATASET.target,
        is_train=True,
        class_names=cfg.DATASET.class_names,
        resize=cfg.DATASET.resize[0],
        keep_difficult=cfg.DATASET.keep_difficult,
        use_mosaic=cfg.DATASET.use_mosaic,
        use_mixup=cfg.DATASET.use_mixup,
        data_type=cfg.DATASET.data_type,
        normalize_box=cfg.DATASET.normalize_box
    )

    testset = create_dataset(
        datadir=cfg.DATASET.datadir,
        target=cfg.DATASET.target,
        is_train=False,
        class_names=cfg.DATASET.class_names,
        resize=cfg.DATASET.resize[0],
        keep_difficult=cfg.DATASET.keep_difficult,
        use_mosaic=cfg.DATASET.use_mosaic,
        use_mixup=cfg.DATASET.use_mixup,
        data_type=cfg.DATASET.data_type,
        normalize_box=cfg.DATASET.normalize_box
    )

    _logger.info('Dataset Size:')
    _logger.info(f"Train: {len(trainset)} - Test: {len(testset)}")

    # build dataloader
    trainloader = create_dataloader(
        dataset=trainset,
        is_train=True,
        batch_size=cfg.DATALOADER.batch_size,
        num_workers=cfg.DATALOADER.num_workers
    )

    _logger.info('Visualize augmentations...')
    if cfg.TRAIN.visualize_aug:
        visualize_augmentations(trainset, cfg, cfg.DATASET.normalize_box)

    # build Model
    model_path = f"task.objectdetection.supervised.models.{cfg.EXP_NAME.split('-')[0].lower()}"
    model = importlib.import_module(
        model_path).build_model(cfg=cfg, device=device)
    model.to(device)
    _logger.info("Total Model Parameters#: {}".format(sum(p.numel()
                 for p in model.parameters())))
    _logger.info("Model A.D. Param#: {}".format(sum(p.numel()
                 for p in model.parameters() if p.requires_grad)))

    # wandb to watch model: gradients, weights and more!
    wandb.watch(model)

    # set training
    # Opimizer & scheduler
    optimizer, optimizer_name, lr = build_optimizer(cfg=cfg, model=model)
    scheduler = build_scheduler(cfg=cfg, optimizer=optimizer, lr=lr)

    # loss
    criterion_path = f"task.objectdetection.supervised.models.{cfg.EXP_NAME.split('-')[0].lower()}.loss"
    criterion, loss_name = importlib.import_module(
        criterion_path).build_loss(cfg=cfg, device=device)

    _logger.info(f"Optimizer: {optimizer_name}")
    _logger.info(f"LR: {lr}")
    _logger.info(f"Loss: {loss_name}")

    # fitting model
    fit_path = model_path = f"task.objectdetection.supervised.models.{cfg.EXP_NAME.split('-')[0].lower()}.tool"
    importlib.import_module(fit_path).fit(model=model,
                                          trainloader=trainloader,
                                          validset=testset,
                                          criterion=criterion,
                                          optimizer=optimizer,
                                          scheduler=scheduler,
                                          epochs=cfg.TRAIN.epochs,
                                          savedir=savedir,
                                          log_interval=cfg.LOG.log_interval,
                                          eval_interval=cfg.LOG.eval_interval,
                                          resume=cfg.RESUME.option,
                                          save_model_path=cfg.RESUME.bestmodel,
                                          vis_test_dir=vis_test_dir,
                                          device=device)

    if cfg.TRAIN.export_to_onnx:
        # build Model
        model_path = f"task.objectdetection.supervised.models.{cfg.EXP_NAME.split('-')[0].lower()}"
        model = importlib.import_module(
            model_path).build_onnx_model(cfg=cfg, device=device)
        model.to(device)
        checkpoint = torch.load(os.path.join(
            savedir, cfg.RESUME.bestmodel), map_location=device)
        _logger.info('loaded weights from {}, best_score {}'.format(
            os.path.join(savedir, cfg.RESUME.bestmodel), checkpoint['best_score']))

        model.load_state_dict(checkpoint['model_state_dict'])

        # export to onnx
        _logger.info('Export to onnx...')
        model.eval()
        # replace nn.SiLU with SiLU
        model = replace_module(model, nn.SiLU, SiLU)

        dummy_input = torch.randn(
            (1, 3, *cfg.DATASET.resize), requires_grad=False).to(device)
        onnx_path = os.path.join(savedir, cfg.RESUME.onnxmodel)
        opset_version = cfg.RESUME.opset_version
        input_names = ["input"]
        output_names = ["output"]
        export_to_onnx(
            model=model,
            dummy_input=dummy_input,
            onnx_path=onnx_path,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names
        )
        wandb.save(os.path.join(
            savedir, cfg.RESUME.onnxmodel), base_path=savedir)
        wandb.save(os.path.join(savedir, 'best_score.json'),
                   base_path=savedir)
        _logger.info('Complete training...')
