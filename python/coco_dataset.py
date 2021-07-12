import torch
from torch.utils.data import Dataset
import os
from pathlib import Path
import numpy as np

from python.homographies import homographic_augmentation, HomographyConfig
from python.netutils import make_points_labels, scale_valid_map


class CocoDataset(Dataset):
    def __init__(self, path, settings, dataset_type, seed=0):
        if dataset_type != 'training' and dataset_type != 'test':
            raise Exception('Incorrect COCO dataset type')
        self.settings = settings
        self.data_path = os.path.join(path, dataset_type)

        files = list(Path(self.data_path).glob('*.*'))
        self.items = [str(file_path) for file_path in files]
        np.random.RandomState(seed).shuffle(self.items)
        self.homography_config = HomographyConfig()

    def __getitem__(self, index):
        item_data = np.load(self.items[index])
        image = torch.from_numpy(item_data['image'])
        points = torch.from_numpy(item_data['points'])

        # take only coordinates
        points = points[:2, :]
        points = torch.transpose(points, 1, 0)
        # swap x and y columns
        points[:, [0, 1]] = points[:, [1, 0]]

        warped_image, warped_points, valid_mask, homography = homographic_augmentation(image, points,
                                                                                       self.homography_config)

        img_h, img_w = image.shape[-2:]
        point_labels = make_points_labels(points.numpy(), img_h, img_w, self.settings.cell)
        warped_point_labels = make_points_labels(warped_points.numpy(), img_h, img_w, self.settings.cell)
        valid_mask = scale_valid_map(valid_mask, img_h, img_w, self.settings.cell)
        
        return image.squeeze(dim=0), torch.from_numpy(point_labels), warped_image.squeeze(dim=0), torch.from_numpy(
            warped_point_labels), valid_mask, homography

    def __len__(self):
        return len(self.items)
