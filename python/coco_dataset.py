import torch
from torch.utils.data import Dataset
import os
from pathlib import Path
import numpy as np

from homographies import homographic_augmentation
from python.netutils import make_points_labels


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
        points = torch.from_numpy(item_data['prob_map'])
        warped_image, warped_points, valid_mask, homography = homographic_augmentation(image, points)

        img_h, img_w = image.shape[-2:]
        point_labels = make_points_labels(points, img_h, img_w, self.settings.cell)
        warped_point_labels = make_points_labels(warped_points, img_h, img_w, self.settings.cell)

        return image, point_labels, warped_image, warped_point_labels, valid_mask, homography

    def __len__(self):
        return len(self.items)
