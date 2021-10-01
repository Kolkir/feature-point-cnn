import torch
from torch.utils.data import Dataset
import os
from pathlib import Path
from numpy.random import default_rng
from src.dataset_utils import read_dataset_item
from src.netutils import make_points_labels


class SyntheticDataset(Dataset):
    def __init__(self, path, settings, dataset_type, seed=0):
        self.settings = settings
        self.data_path = os.path.join(path, dataset_type)

        files = list(Path(self.data_path).glob('*.*'))
        self.items = [str(file_path) for file_path in files]
        default_rng(seed).shuffle(self.items)

    def __getitem__(self, index):
        file_name = self.items[index]
        image, points = read_dataset_item(file_name)
        img_h, img_w = image.shape[1:]
        points_map = make_points_labels(points.numpy(), img_h, img_w, self.settings.cell)
        return image, torch.from_numpy(points_map)

    def __len__(self):
        return len(self.items)
