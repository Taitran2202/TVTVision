import os
import wandb
import warnings
import logging
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from .model import STFPM
from .tool import training
from .loss import STFPMLoss
from ...plot_image import plot_image
from ...data import create_dataset, create_dataloader
from utils.export_onnx import export_to_onnx
from utils.seed import seeding
from utils.log import setup_default_logging
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler
_logger = logging.getLogger('train')
warnings.filterwarnings("ignore")


def visualize_augmentations(dataset, samples=2):
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for j, (image, target) in enumerate(loader):
        if j == samples:
            break
        plot_image(image[0].permute(1, 2, 0), target)


def run(cfg):
    # savedir
    cfg.EXP_NAME = cfg.EXP_NAME + f"-{cfg.MODEL.backbone}"
    savedir = os.path.join(cfg.RESULT.savedir, cfg.TASK,
                           cfg.METHOD, cfg.DATASET.target, cfg.EXP_NAME)
    os.makedirs(savedir, exist_ok=True)

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
    _logger.info(f"Steps: {cfg.TRAIN.num_training_steps}")

    # build datasets
    _logger.info('Setting up data...')
    trainset = create_dataset(
        datadir=cfg.DATASET.datadir,
        target=cfg.DATASET.target,
        is_train=True,
        resize=cfg.DATASET.resize,
    )

    testset = create_dataset(
        datadir=cfg.DATASET.datadir,
        target=cfg.DATASET.target,
        is_train=False,
        resize=cfg.DATASET.resize
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

    testloader = create_dataloader(
        dataset=testset,
        is_train=False,
        batch_size=cfg.DATALOADER.batch_size,
        num_workers=cfg.DATALOADER.num_workers
    )

    _logger.info('Visualize augmentations...')
    if cfg.TRAIN.visualize_aug:
        visualize_augmentations(trainset, cfg.TRAIN.num_sample)

    # build STFPM
    model = STFPM(
        layers=cfg.MODEL.layers,
        input_size=cfg.DATASET.resize,
        backbone=cfg.MODEL.backbone,
        use_feature_pooler=cfg.MODEL.use_feature_pooler,
        use_tiler=cfg.MODEL.use_tiler,
        tiler_size=cfg.MODEL.tiler_size,
        stride=cfg.MODEL.stride,
    )
    model.to(device)

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
    m_criterion = STFPMLoss()

    loss_name = "STFPM Loss"
    _logger.info(f"Optimizer: {optimizer_name}")
    _logger.info(f"LR: {lr}")
    _logger.info(f"Loss: {loss_name}")

    # Fitting model
    training(
        model=model,
        trainloader=trainloader,
        validloader=testloader,
        criterion=m_criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_training_steps=cfg.TRAIN.num_training_steps,
        loss_weights=1,
        log_interval=cfg.LOG.log_interval,
        eval_interval=cfg.LOG.eval_interval,
        savedir=savedir,
        resume=cfg.RESUME.option,
        save_model_path=cfg.RESUME.bestmodel,
        use_wandb=cfg.TRAIN.use_wandb,
        top_k=cfg.TRAIN.top_k,
        compute_threshold=cfg.TRAIN.compute_threshold,
        beta=cfg.TRAIN.beta,
        device=device
    )

    if cfg.TRAIN.export_to_onnx:
        checkpoint = torch.load(os.path.join(
            savedir, cfg.RESUME.bestmodel), map_location=device)
        _logger.info('loaded weights from {}, step {}, best_score {}'.format(
            os.path.join(savedir, cfg.RESUME.bestmodel), checkpoint['step'], checkpoint['best_score']))

        model.load_state_dict(checkpoint['model_state_dict'])

        # export to onnx
        _logger.info('Export to onnx...')
        model.eval()
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
        wandb.save(os.path.join(savedir, cfg.RESUME.onnxmodel),
                   base_path=savedir)
        wandb.save(os.path.join(savedir, 'best_score.json'),
                   base_path=savedir)
        _logger.info('Complete training...')