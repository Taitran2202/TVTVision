import os
from typing import Tuple
from .module.crnnbasedataset import CRNNBaseDataset


class Synth90kDataset(CRNNBaseDataset):
    def __init__(
        self, datadir: str, target: str, train: bool, chars: str,
        resize: Tuple[int, int] = (512, 512),
    ):
        super(Synth90kDataset, self).__init__(
            datadir, target, train, chars, resize)

    def load_from_raw_files(self, root_dir):
        mapping = {}
        with open(os.path.join(root_dir, 'lexicon.txt')) as f:
            mapping = {i: line.strip() for i, line in enumerate(f)}

        paths_file = 'annotation_train.txt' if self.train else 'annotation_val.txt'
        paths, texts = [], []

        with open(os.path.join(root_dir, paths_file)) as f:
            for i, line in enumerate(f):
                if paths_file == 'annotation_val.txt' and i > 5000:
                    break
                path, index_str = line.strip().split(' ')
                path = os.path.join(root_dir, path)
                index = int(index_str)
                text = mapping[index]
                paths.append(path)
                texts.append(text)

        return paths, texts
