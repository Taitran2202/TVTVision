from typing import Tuple
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class PSEBaseDataset(Dataset):
    def __init__(
        self, datadir: str, target: str, is_train: bool, aug,
        resize: Tuple[int, int] = (512, 512), short_size: int = 736,
        kernel_num: int = 7, min_scale: float = 0.7
    ):
        super(PSEBaseDataset, self).__init__()
        self.datadir = datadir
        self.target = target
        self.is_train = is_train
        self.resize = resize
        self.short_size = short_size
        self.kernel_num = kernel_num
        self.min_scale = min_scale
        self.aug = aug

        # convert ndarray into tensor
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.max_word_num = 200

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if self.is_train:
            image, gt_text, gt_kernels, training_mask = self.make_gt_score(
                index)

            gt_text = torch.from_numpy(gt_text).long()
            gt_kernels = torch.from_numpy(gt_kernels).long()
            training_mask = torch.from_numpy(training_mask).long()

            data = dict(
                imgs=self.image_transform(image),
                gt_texts=gt_text,
                gt_kernels=gt_kernels,
                training_masks=training_mask,
            )

            return data
        else:
            img_path, single_img_bboxes = self.load_test_dataset_iou(index)

            return img_path, single_img_bboxes

    def load_test_dataset_iou(self, index):
        if self.target == 'synth':
            total_img_path, total_bboxes_gt = self.load_synthtext_gt(index)
        elif self.target == 'icdar17':
            total_img_path, total_bboxes_gt = self.load_icdar2017_gt(index)
        elif self.target == 'icdar15':
            total_img_path, total_bboxes_gt = self.load_icdar2015_gt(index)
        elif self.target == 'icdar13':
            total_img_path, total_bboxes_gt = self.load_icdar2013_gt(index)

        return total_img_path, total_bboxes_gt
