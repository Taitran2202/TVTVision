from torch.utils.data import DataLoader
from .icdardataset import EASTICDARDataset
from .synthdataset import EASTSynthTextDataSet


def create_synth_dataset(
    datadir: str, target: str, is_train: bool,
    scale: float = 0.25, length: int = 512
):
    dataset = EASTSynthTextDataSet(
        datadir=datadir,
        target=target,
        is_train=is_train,
        scale=scale,
        length=length,
    )

    return dataset


def create_icdar_dataset(
    datadir: str, target: str, is_train: bool,
    scale: float = 0.25, length: int = 512
):
    dataset = EASTICDARDataset(
        datadir=datadir,
        target=target,
        is_train=is_train,
        scale=scale,
        length=length,
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
