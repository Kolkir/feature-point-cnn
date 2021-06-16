import torch
from torch.utils.data import Dataset
import os
from pathlib import Path
import numpy as np


class CocoDataset(Dataset):
    def __init__(self, path, settings, dataset_type, seed=0):
        if dataset_type != 'training' and dataset_type != 'test':
            raise Exception('Incorrect synthetic dataset type')
        self.settings = settings
        self.data_path = os.path.join(path, dataset_type)

        files = list(Path(self.data_path).glob('*.*'))
        files.sort()
        self.items = [str(file_path) for file_path in self.items]
        np.random.RandomState(seed).shuffle(self.items)

    def __getitem__(self, index):
        item_data = np.load(self.items[index])
        image = item_data['image']
        points = item_data['points']
        return torch.from_numpy(image), torch.from_numpy(points)

    def __len__(self):
        return len(self.items)
