import torch
from torch.utils.data import Dataset
import os
from pathlib import Path
import numpy as np

from homographies import homographic_augmentation


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
        image = torch.from_numpy(item_data['image'])
        points = torch.from_numpy(item_data['points'])
        warped_image, warped_points, valid_mask = homographic_augmentation(image, points)
        return image, points, warped_image, warped_points, valid_mask

    def __len__(self):
        return len(self.items)
