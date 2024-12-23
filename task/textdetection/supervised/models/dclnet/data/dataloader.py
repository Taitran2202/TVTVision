from torch.utils.data import DataLoader
from .icdardataset import DCLNetICDARDataset
from .synthdataset import DCLNetSynthTextDataSet


def create_synth_dataset(
    datadir: str, target: str, is_train: bool,
    scale: float = 0.25, length: int = 512,
    min_side: int = 640, max_side: int = 1920
):
    dataset = DCLNetSynthTextDataSet(
        datadir=datadir,
        target=target,
        is_train=is_train,
        scale=scale,
        length=length,
        min_side=min_side,
        max_side=max_side
    )

    return dataset


def create_icdar_dataset(
    datadir: str, target: str, is_train: bool,
    scale: float = 0.25, length: int = 512,
    min_side: int = 640, max_side: int = 1920
):
    dataset = DCLNetICDARDataset(
        datadir=datadir,
        target=target,
        is_train=is_train,
        scale=scale,
        length=length,
        min_side=min_side,
        max_side=max_side
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
