from typing import Tuple
from torch.utils.data import DataLoader
from .synthdataset import SynthTextDataSet
from .icdardataset import ICDARDataset


def create_synth_dataset(
    datadir: str, target: str, is_train: bool, aug,
    resize: Tuple[int, int] = (256, 256),
    gauss_init_size: int = 200, gauss_sigma: int = 40,
    enlarge_region: Tuple[float, float] = [0.5, 0.5],  # x axis, y axis,
    enlarge_affinity: Tuple[float, float] = [0.5, 0.5]  # x axis, y axis
):
    dataset = SynthTextDataSet(
        datadir=datadir,
        target=target,
        is_train=is_train,
        aug=aug,
        resize=resize,
        gauss_init_size=gauss_init_size,
        gauss_sigma=gauss_sigma,
        enlarge_region=enlarge_region,
        enlarge_affinity=enlarge_affinity
    )

    return dataset


def create_icdar_dataset(
    datadir: str, target: str, is_train: bool, aug, vis_train_dir: str,
    pseudo_vis_opt: bool, net,  do_not_care_label: Tuple[str, str] = ['###', ''], device: str = 'cpu',
    resize: Tuple[int, int] = (768, 768), gauss_init_size: int = 200,
    gauss_sigma: int = 40, enlarge_region: Tuple[float, float] = [0.5, 0.5],
    enlarge_affinity: Tuple[float, float] = [0.5, 0.5]
):
    dataset = ICDARDataset(
        datadir=datadir,
        target=target,
        is_train=is_train,
        aug=aug,
        vis_train_dir=vis_train_dir,
        pseudo_vis_opt=pseudo_vis_opt,
        net=net,
        do_not_care_label=do_not_care_label,
        device=device,
        resize=resize,
        gauss_init_size=gauss_init_size,
        gauss_sigma=gauss_sigma,
        enlarge_region=enlarge_region,
        enlarge_affinity=enlarge_affinity
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
