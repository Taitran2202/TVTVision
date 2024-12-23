from typing import Tuple
from torch.utils.data import DataLoader
from .synthdataset import PANSynthTextDataSet
from .icdardataset import PANICDARDataset


def create_synth_dataset(
    datadir: str, target: str, is_train: bool, aug,
    resize: Tuple[int, int] = (512, 512), short_size: int = 736,
    kernel_scale: float = 0.5
):
    dataset = PANSynthTextDataSet(
        datadir=datadir,
        target=target,
        is_train=is_train,
        aug=aug,
        resize=resize,
        short_size=short_size,
        kernel_scale=kernel_scale
    )

    return dataset


def create_icdar_dataset(
    datadir: str, target: str, is_train: bool, aug,
    resize: Tuple[int, int] = (512, 512), short_size: int = 736,
    kernel_scale: float = 0.5
):
    dataset = PANICDARDataset(
        datadir=datadir,
        target=target,
        is_train=is_train,
        aug=aug,
        resize=resize,
        short_size=short_size,
        kernel_scale=kernel_scale
    )

    return dataset


def create_dataloader(dataset, is_train: bool, batch_size: int = 8, num_workers: int = 2):
    dataloader = DataLoader(
        dataset,
        shuffle=is_train,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return dataloader
