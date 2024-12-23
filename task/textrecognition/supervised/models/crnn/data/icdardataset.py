import os
from typing import Tuple
from .module.crnnbasedataset import CRNNBaseDataset


class ICDARDataset(CRNNBaseDataset):
    def __init__(
        self, datadir: str, target: str, train: bool, chars: str,
        resize: Tuple[int, int] = (512, 512)
    ):
        super(ICDARDataset, self).__init__(
            datadir, target, train, chars, resize)

    def load_from_raw_files(self, root_dir):
        path_file = 'train' if self.train else 'test'

        with open(os.path.join(root_dir, path_file, 'gt.txt'), 'r') as f:
            paths_texts = [line.strip().split(', ') for line in f]
            paths = []
            texts = []

            for p, t in paths_texts:
                # t = t.replace('"', '').lower()
                t = t.replace('"', '')
                if all(c in self.chars for c in t):
                    paths.append(os.path.join(root_dir, path_file, p))
                    texts.append(t)

        return paths, texts
