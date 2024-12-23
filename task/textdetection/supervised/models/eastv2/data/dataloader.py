from typing import Tuple
from torch.utils.data import DataLoader
from .icdardataset import EASTv2ICDARDataset
from .synthdataset import EASTv2SynthTextDataSet


def create_synth_dataset(
    datadir: str, target: str, is_train: bool,
    do_not_care_label: Tuple[str, str] = ['###', ''], min_text_size: float = 10,
    background_ratio: float = 0.125, min_crop_side_ratio: float = 0.1,
    img_size: int = 512, random_scale: list[float] = [0.5, 1, 2.0, 3.0]
):
    dataset = EASTv2SynthTextDataSet(
        datadir=datadir,
        target=target,
        is_train=is_train,
        do_not_care_label=do_not_care_label,
        background_ratio=background_ratio,
        min_text_size=min_text_size,
        min_crop_side_ratio=min_crop_side_ratio,
        img_size=img_size,
        random_scale=random_scale
    )

    return dataset


def create_icdar_dataset(
    datadir: str, target: str, is_train: bool,
    do_not_care_label: Tuple[str, str] = ['###', ''], min_text_size: float = 10,
    background_ratio: float = 0.125, min_crop_side_ratio: float = 0.1,
    img_size: int = 512, random_scale: list[float] = [0.5, 1, 2.0, 3.0]
):
    dataset = EASTv2ICDARDataset(
        datadir=datadir,
        target=target,
        is_train=is_train,
        do_not_care_label=do_not_care_label,
        background_ratio=background_ratio,
        min_text_size=min_text_size,
        min_crop_side_ratio=min_crop_side_ratio,
        img_size=img_size,
        random_scale=random_scale
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
