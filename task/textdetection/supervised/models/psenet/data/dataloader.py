from typing import Tuple
from torch.utils.data import DataLoader
from .synthdataset import PSESynthTextDataSet
from .icdardataset import PSEICDARDataset


def create_synth_dataset(
    datadir: str, target: str, is_train: bool, aug,
    resize: Tuple[int, int] = (512, 512), short_size: int = 736,
    kernel_num: int = 7, min_scale: float = 0.7
):
    dataset = PSESynthTextDataSet(
        datadir=datadir,
        target=target,
        is_train=is_train,
        aug=aug,
        resize=resize,
        short_size=short_size,
        kernel_num=kernel_num,
        min_scale=min_scale
    )

    return dataset


def create_icdar_dataset(
    datadir: str, target: str, is_train: bool, aug,
    resize: Tuple[int, int] = (512, 512), short_size: int = 736,
    kernel_num: int = 7, min_scale: float = 0.7
):
    dataset = PSEICDARDataset(
        datadir=datadir,
        target=target,
        is_train=is_train,
        aug=aug,
        resize=resize,
        short_size=short_size,
        kernel_num=kernel_num,
        min_scale=min_scale
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
