from typing import Tuple
from torch.utils.data import DataLoader
from .dataset import MVTecDataset


def create_dataset(
    datadir: str, target: str, is_train: bool,
    resize: Tuple[int, int] = (224, 224), transform=None
):
    dataset = MVTecDataset(
        datadir=datadir,
        target=target,
        is_train=is_train,
        resize=resize,
        transform=transform
    )

    return dataset


def create_dataloader(dataset, is_train: bool, batch_size: int = 16, num_workers: int = 2):
    dataloader = DataLoader(
        dataset,
        shuffle=is_train,
        batch_size=batch_size,
        num_workers=num_workers
    )

    return dataloader
