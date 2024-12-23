from typing import Tuple
from torch.utils.data import DataLoader
from .dataset import HLSafeCountDataset


def create_hlsafecount_dataset(
    datadir: str, target: str, is_train: bool, aug, num_boxes: int,
    num_train: float = 0.9, resize: Tuple[int, int] = (256, 256),
):
    dataset = HLSafeCountDataset(
        datadir=datadir,
        target=target,
        is_train=is_train,
        aug=aug,
        num_boxes=num_boxes,
        num_train=num_train,
        resize=resize,
    )

    return dataset


def create_hlsafecount_dataloader(dataset, is_train: bool, batch_size: int = 1, num_workers: int = 2):
    dataloader = DataLoader(
        dataset,
        shuffle=is_train,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return dataloader
