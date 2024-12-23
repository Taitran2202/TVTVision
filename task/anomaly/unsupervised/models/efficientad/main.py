import os
import numpy as np
import wandb
import warnings
import logging
from omegaconf import OmegaConf
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from .model import EfficientAD
from .tool import training
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


class TransformsWrapper:
    def __init__(self, t: A.Compose):
        self.transforms = t

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))


def prepare_imagenette_data(imagenet_dir, image_size, batch_size):
    data_transforms_imagenet = A.Compose(
        [  # We obtain an image P ∈ R 3×256×256 from ImageNet by choosing a random image,
            # resizing it to 512 × 512,
            A.Resize(image_size[0] * 2, image_size[1] * 2),
            # converting it to gray scale with a probability of 0.3
            A.ToGray(p=0.3),
            # and cropping the center 256 × 256 pixels
            A.CenterCrop(image_size[0], image_size[1]),
            A.ToFloat(always_apply=False, p=1.0, max_value=255),
            ToTensorV2(),
        ]
    )

    imagenet_dataset = ImageFolder(
        imagenet_dir, transform=TransformsWrapper(t=data_transforms_imagenet))
    imagenet_loader = DataLoader(
        imagenet_dataset, batch_size=batch_size, shuffle=True)

    return imagenet_loader


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
    full_train_set = create_dataset(
        datadir=cfg.DATASET.datadir,
        target=cfg.DATASET.target,
        is_train=True,
        resize=cfg.DATASET.resize,
    )
    num_samples = len(full_train_set)
    num_train = int(num_samples * cfg.DATASET.num_train)
    num_validation = num_samples - num_train

    # split trainset, validset
    trainset, validationset = random_split(full_train_set, [
        num_train, num_validation], generator=torch.Generator().manual_seed(42))

    testset = create_dataset(
        datadir=cfg.DATASET.datadir,
        target=cfg.DATASET.target,
        is_train=False,
        resize=cfg.DATASET.resize,
    )

    _logger.info('Dataset Size:')
    _logger.info(
        f"Train: {len(trainset)} - Valid: {len(validationset)} - Test: {len(testset)}")

    # build dataloader
    trainloader = create_dataloader(
        dataset=trainset,
        is_train=True,
        batch_size=cfg.DATALOADER.batch_size,
        num_workers=cfg.DATALOADER.num_workers
    )

    validloader = create_dataloader(
        dataset=validationset,
        is_train=False,
        batch_size=cfg.DATALOADER.batch_size,
        num_workers=cfg.DATALOADER.num_workers
    )

    testloader = create_dataloader(
        dataset=testset,
        is_train=False,
        batch_size=cfg.DATALOADER.batch_size,
        num_workers=cfg.DATALOADER.num_workers
    )

    penaltyloader = prepare_imagenette_data(
        imagenet_dir=cfg.DATASET.datadir_penalty,
        image_size=cfg.DATASET.resize,
        batch_size=cfg.DATALOADER.batch_size
    )

    _logger.info('Visualize augmentations...')
    if cfg.TRAIN.visualize_aug:
        visualize_augmentations(trainset, cfg.TRAIN.num_sample)

    # build EfficientAD
    model = EfficientAD(
        model_size=cfg.MODEL.backbone,
        input_size=cfg.DATASET.resize,
        teacher_out_channels=cfg.MODEL.teacher_out_channels,
        padding=cfg.MODEL.padding,
        pad_maps=cfg.MODEL.pad_maps,
    )
    model.to(device)
    model.mean_std.update(model.teacher_channel_mean_std(trainloader))

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
    loss_name = "EfficientAD Loss"
    _logger.info(f"Optimizer: {optimizer_name}")
    _logger.info(f"LR: {lr}")
    _logger.info(f"Loss: {loss_name}")

    # Fitting model
    training(
        model=model,
        trainloader=trainloader,
        validloader=validloader,
        testloader=testloader,
        penaltyloader=penaltyloader,
        criterion=None,
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
        output_names = ["output1", "output2", "output3"]
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
