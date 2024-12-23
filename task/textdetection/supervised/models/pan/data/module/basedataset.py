from typing import Tuple
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .utils import get_vocabulary


class PANBaseDataset(Dataset):
    def __init__(
        self, datadir: str, target: str, is_train: bool, aug,
        resize: Tuple[int, int] = (512, 512), short_size: int = 736,
        kernel_scale: float = 0.5
    ):
        super(PANBaseDataset, self).__init__()
        self.datadir = datadir
        self.target = target
        self.is_train = is_train
        self.resize = resize
        self.short_size = short_size
        self.kernel_scale = kernel_scale
        self.aug = aug

        self.voc, self.char2id, self.id2char = get_vocabulary('LOWERCASE')
        self.max_word_num = 200
        self.max_word_len = 32

        # convert ndarray into tensor
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if self.is_train:
            image, gt_text, gt_kernels, training_mask, gt_instance, gt_bboxes, gt_words, word_mask = self.make_gt_score(
                index)

            gt_text = torch.from_numpy(gt_text).long()
            gt_kernels = torch.from_numpy(gt_kernels).long()
            training_mask = torch.from_numpy(training_mask).long()
            gt_instance = torch.from_numpy(gt_instance).long()
            gt_bboxes = torch.from_numpy(gt_bboxes).long()
            gt_words = torch.from_numpy(gt_words).long() if gt_words else None
            word_mask = torch.from_numpy(word_mask).long() if word_mask else None
            
            if gt_words is None or word_mask is None:
                data = dict(
                    imgs=self.image_transform(image),
                    gt_texts=gt_text,
                    gt_kernels=gt_kernels,
                    training_masks=training_mask,
                    gt_instances=gt_instance,
                    gt_bboxes=gt_bboxes
                )
            else:
                data = dict(
                    imgs=self.image_transform(image),
                    gt_texts=gt_text,
                    gt_kernels=gt_kernels,
                    training_masks=training_mask,
                    gt_instances=gt_instance,
                    gt_bboxes=gt_bboxes,
                    gt_words=gt_words,
                    word_masks=word_mask
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
