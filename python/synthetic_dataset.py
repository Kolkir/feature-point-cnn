import torch
from torch.utils.data import Dataset
import os
from pathlib import Path
import numpy as np
import torchvision
from netutils import make_points_labels, get_points_coordinates


def read_image(filename):
    image = torchvision.io.read_image(filename, torchvision.io.image.ImageReadMode.GRAY)
    image = image.to(dtype=torch.float32) / 255.
    return image


def read_points(filename, cell_size, img_h, img_w):
    points = np.load(filename).astype(np.float32)
    points_map = make_points_labels(points, img_h, img_w, cell_size)
    return torch.from_numpy(points_map)


class SyntheticDataset(Dataset):
    def __init__(self, path, settings, dataset_type, seed=0):
        if dataset_type != 'training' and dataset_type != 'validation' and dataset_type != 'test':
            raise Exception('Incorrect synthetic dataset type')
        self.settings = settings
        self.images_path = os.path.join(path, 'images', dataset_type)
        self.points_path = os.path.join(path, 'points', dataset_type)

        image_files = list(Path(self.images_path).glob('*.*'))
        image_files.sort()
        point_files = list(Path(self.points_path).glob('*.*'))
        point_files.sort()
        self.items = list(zip(image_files, point_files))
        self.items = [(str(img_path), str(pts_path)) for img_path, pts_path in self.items]
        np.random.RandomState(seed).shuffle(self.items)

    def __getitem__(self, index):
        image = read_image(self.items[index][0])
        img_h = image.shape[1]
        img_w = image.shape[2]
        points = read_points(self.items[index][1], self.settings.cell, img_h, img_w)
        return image, points

    def __len__(self):
        return len(self.items)
