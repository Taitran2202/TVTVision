from typing import Tuple
from .dataset import EdgeDataset
from torch.utils.data import DataLoader


def create_dataset(
    datadir: str, target: str, is_train: bool, num_train: float = 0.9,
    threshold: float = 0.3, resize: Tuple[int, int] = (224, 224),
):
    dataset = EdgeDataset(
        datadir=datadir,
        target=target,
        is_train=is_train,
        num_train=num_train,
        threshold=threshold,
        resize=resize,
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
