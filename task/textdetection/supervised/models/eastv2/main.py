import os
import wandb
import warnings
import logging
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset
from .model import EASTv2
from .loss import EASTv2Loss
from .tool import fit
from .plot_image import plot_image
from .data import create_synth_dataset, create_icdar_dataset, create_dataloader
from utils.export_onnx import export_to_onnx
from utils.seed import seeding
from utils.log import setup_default_logging
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler
_logger = logging.getLogger('train')
warnings.filterwarnings("ignore")


def visualize_augmentations(dataset, samples=2):
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for j, (image, score_map, _, training_mask) in enumerate(loader):
        if j == samples:
            break
        plot_image(image[0].permute(1, 2, 0),
                   score_map[0].squeeze(), training_mask[0].squeeze())


def run(cfg):
    # savedir
    cfg.EXP_NAME = cfg.EXP_NAME + f"-{cfg.MODEL.backbone}"
    savedir = os.path.join(cfg.RESULT.savedir, cfg.TASK,
                           cfg.METHOD, cfg.DATASET.target, cfg.EXP_NAME)
    vis_test_dir = os.path.join(cfg.RESULT.savedir, cfg.TASK, cfg.METHOD,
                                cfg.DATASET.target, cfg.EXP_NAME, cfg.RESULT.vis_test_dir)
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
    _logger.info(f"Image Size: {cfg.DATASET.img_size}")
    _logger.info(f"Batch Size: {cfg.DATALOADER.batch_size}")
    _logger.info(f"Epochs: {cfg.TRAIN.epochs}")

    # build datasets
    _logger.info('Setting up data...')
    if cfg.DATASET.target == 'synth':
        trainset = create_synth_dataset(
            datadir=cfg.DATASET.datadir,
            target=cfg.DATASET.target,
            is_train=True,
            do_not_care_label=cfg.DATASET.do_not_care_label,
            background_ratio=cfg.DATASET.background_ratio,
            img_size=cfg.DATASET.img_size[0],
            random_scale=cfg.DATASET.random_scale,
            min_text_size=cfg.DATASET.min_text_size,
            min_crop_side_ratio=cfg.DATASET.min_crop_side_ratio
        )
    elif 'icdar' in cfg.DATASET.target:
        trainset = create_icdar_dataset(
            datadir=cfg.DATASET.datadir,
            target=cfg.DATASET.target,
            is_train=True,
            do_not_care_label=cfg.DATASET.do_not_care_label,
            background_ratio=cfg.DATASET.background_ratio,
            img_size=cfg.DATASET.img_size[0],
            random_scale=cfg.DATASET.random_scale,
            min_text_size=cfg.DATASET.min_text_size,
            min_crop_side_ratio=cfg.DATASET.min_crop_side_ratio
        )

    if cfg.DATASET.test_folder == 'synth':
        testset = create_synth_dataset(
            datadir=cfg.DATASET.datadir,
            target=cfg.DATASET.test_folder,
            is_train=False,
            do_not_care_label=cfg.DATASET.do_not_care_label,
            background_ratio=cfg.DATASET.background_ratio,
            img_size=cfg.DATASET.img_size[0],
            random_scale=cfg.DATASET.random_scale,
        )
    elif 'icdar' in cfg.DATASET.test_folder:
        testset = create_icdar_dataset(
            datadir=cfg.DATASET.datadir,
            target=cfg.DATASET.test_folder,
            is_train=False,
            do_not_care_label=cfg.DATASET.do_not_care_label,
            background_ratio=cfg.DATASET.background_ratio,
            img_size=cfg.DATASET.img_size[0],
            random_scale=cfg.DATASET.random_scale,
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
        visualize_augmentations(trainset, cfg.TRAIN.num_sample)

    # build EASTv2
    model = EASTv2(
        backbone=cfg.MODEL.backbone,
        pretrained=cfg.MODEL.pretrained,
        inner_channels=cfg.MODEL.inner_channels,
        scope=cfg.DATASET.img_size[0]
    )
    model.to(device)

    # if cfg.TRAIN.fine_tuning:
    #     _logger.info("Load weight model to fine tuning")
    #     checkpoint = torch.load(
    #         cfg.TRAIN.weights_pretrained, map_location=device)
    #     model.load_state_dict(checkpoint['model_state_dict'])

    _logger.info("Total Model Parameters#: {}".format(sum(p.numel()
                 for p in model.parameters())))
    _logger.info("Model A.D. Param#: {}".format(sum(p.numel()
                 for p in model.parameters() if p.requires_grad)))

    # wandb to watch model: gradients, weights and more!
    wandb.watch(model)

    # Set training
    # Opimizer & scheduler
    optimizer, optimizer_name, lr = build_optimizer(cfg=cfg, model=model)
    scheduler = build_scheduler(cfg=cfg, optimizer=optimizer, lr=lr)

    # loss
    east_criterion = EASTv2Loss()
    loss_name = "EAST Loss"
    _logger.info(f"Optimizer: {optimizer_name}")
    _logger.info(f"LR: {lr}")
    _logger.info(f"Loss: {loss_name}")

    # Fitting model
    fit(
        model=model,
        trainloader=trainloader,
        testdataset=testset,
        criterion=east_criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=cfg.TRAIN.epochs,
        loss_weights=1,
        score_thresh=cfg.TEST.score_thresh,
        nms_thresh=cfg.TEST.nms_thresh,
        cover_thresh=cfg.TEST.cover_thresh,
        resize=cfg.DATASET.img_size,
        log_interval=cfg.LOG.log_interval,
        eval_interval=cfg.LOG.eval_interval,
        savedir=savedir,
        resume=cfg.RESUME.option,
        save_model_path=cfg.RESUME.bestmodel,
        vis_test_dir=vis_test_dir,
        device=device
    )

    if cfg.TRAIN.export_to_onnx:
        checkpoint = torch.load(os.path.join(
            savedir, cfg.RESUME.bestmodel), map_location=device)
        _logger.info('loaded weights from {}, best_hmean {}'.format(
            os.path.join(savedir, cfg.RESUME.bestmodel), checkpoint['best_score']))

        model.load_state_dict(checkpoint['model_state_dict'])

        # export to onnx
        _logger.info('Export to onnx...')
        model.eval()
        dummy_input = torch.randn(
            (1, 3, *cfg.DATASET.resize), requires_grad=False).to(device)
        onnx_path = os.path.join(savedir, cfg.RESUME.onnxmodel)
        opset_version = cfg.RESUME.opset_version
        input_names = ["input"]
        output_names = ["output1", "output2"]
        export_to_onnx(
            model=model,
            dummy_input=dummy_input,
            onnx_path=onnx_path,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names
        )
        wandb.save(os.path.join(savedir, cfg.RESUME.onnxmodel),
                   base_path=savedir)
        wandb.save(os.path.join(savedir, 'best_score.json'),
                   base_path=savedir)
        _logger.info('Complete training...')
