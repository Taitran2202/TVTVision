from typing import Tuple
import torch
from torch.utils.data import DataLoader
from .synth90kdataset import Synth90kDataset
from .icdardataset import ICDARDataset


def create_synth90k_dataset(
    datadir: str, target: str, train: bool, chars: str,
    resize: Tuple[int, int] = (256, 256),
):
    dataset = Synth90kDataset(
        datadir=datadir,
        target=target,
        train=train,
        chars=chars,
        resize=resize
    )

    return dataset


def create_icdar_dataset(
    datadir: str, target: str, train: bool, chars: str,
    resize: Tuple[int, int] = (256, 256),
):
    dataset = ICDARDataset(
        datadir=datadir,
        target=target,
        train=train,
        chars=chars,
        resize=resize
    )

    return dataset


def create_dataloader(dataset, train: bool, batch_size: int = 8, num_workers: int = 2):
    dataloader = DataLoader(
        dataset,
        shuffle=train,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return dataloader


def collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths
