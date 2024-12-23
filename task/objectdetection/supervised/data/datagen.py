import torch
from torch.utils.data import DataLoader
from .dataset import VocDataset, YOLODataset, SynthDataset
from .module import YoloAugmentation, YoloColorAugmentation, DETRAugmentation, DETRColorAugmentation


def create_dataset(
    datadir: str, target: str, is_train: bool, class_names, resize: int = 640, keep_difficult: bool = False,
    use_mosaic: bool = False, use_mixup: bool = False, data_type: str = "yolo", normalize_box: bool = False,
):
    if data_type == "voc":
        dataset = VocDataset(
            datadir=datadir,
            target=target,
            is_train=is_train,
            class_names=class_names,
            image_resize=resize,
            keep_difficult=keep_difficult,
            use_mosaic=use_mosaic,
            use_mixup=use_mixup,
            transformer=DETRAugmentation(
                resize, p=0.5) if normalize_box else YoloAugmentation(resize, p=0.5),
            color_transformer=DETRColorAugmentation(
                resize, p=0.5) if normalize_box else YoloColorAugmentation(resize, p=0.5)
        )
    elif data_type == "yolo":
        dataset = YOLODataset(
            datadir=datadir,
            target=target,
            is_train=is_train,
            class_names=class_names,
            image_resize=resize,
            keep_difficult=keep_difficult,
            use_mosaic=use_mosaic,
            use_mixup=use_mixup,
            transformer=DETRAugmentation(
                resize, p=0.5) if normalize_box else YoloAugmentation(resize, p=0.5),
            color_transformer=DETRColorAugmentation(
                resize, p=0.5) if normalize_box else YoloColorAugmentation(resize, p=0.5)
        )
    elif data_type == "synth":
        dataset = SynthDataset(
            datadir=datadir,
            target=target,
            is_train=is_train,
            class_names=class_names,
            image_resize=resize,
            keep_difficult=keep_difficult,
            use_mosaic=use_mosaic,
            use_mixup=use_mixup,
            transformer=DETRAugmentation(
                resize, p=0.5) if normalize_box else YoloAugmentation(resize, p=0.5),
            color_transformer=DETRColorAugmentation(
                resize, p=0.5) if normalize_box else YoloColorAugmentation(resize, p=0.5)
        )
    else:
        raise ValueError("Data type must be 'voc' or 'yolo'")

    return dataset


def create_dataloader(dataset, is_train: bool, batch_size: int = 16, num_workers: int = 2):
    dataloader = DataLoader(
        dataset,
        shuffle=is_train,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    return dataloader


def collate_fn(batch):
    targets = []
    images = []

    for sample in batch:
        image = sample[0]
        target = sample[1]

        target_boxes = [torch.tensor(box) for box in target['boxes']]
        target_labels = [torch.tensor(label) for label in target['labels']]

        if target_boxes and target_labels:
            target['boxes'] = torch.stack(target_boxes).to(torch.float32)
            target['labels'] = torch.stack(target_labels).to(torch.int64)
        else:
            target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
            target['labels'] = torch.empty((0,), dtype=torch.int64)

        images.append(image)
        targets.append(target)

    images = torch.stack(images)  # [B, C, H, W]

    return images, targets
