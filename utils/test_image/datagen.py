from .dataset import TestDataset


def create_test_dataset(
    cfg, datadir: str
):
    dataset = TestDataset(
        cfg=cfg,
        datadir=datadir
    )

    return dataset
