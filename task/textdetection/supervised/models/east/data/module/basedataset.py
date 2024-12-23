from torch.utils.data import Dataset
import torchvision.transforms as transforms


class EASTBaseDataset(Dataset):
    def __init__(
        self, datadir: str, target: str, is_train: bool,
        scale: float = 0.25, length: int = 512
    ):
        super(EASTBaseDataset, self).__init__()
        self.datadir = datadir
        self.target = target
        self.is_train = is_train
        self.scale = scale
        self.length = length

        # convert ndarray into tensor
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if self.is_train:
            image, score_map, geo_map, ignored_map = self.make_gt_score(index)

            return self.image_transform(image), score_map, geo_map, ignored_map
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
