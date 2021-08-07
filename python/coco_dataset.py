import torch
from torch.utils.data import Dataset
import os
from pathlib import Path
import numpy as np
from numpy.random import default_rng

from dataset_transforms import dataset_transforms
from homographies import homographic_augmentation, HomographyConfig
from netutils import make_points_labels, scale_valid_map


class CocoDataset(Dataset):
    def __init__(self, path, settings, dataset_type, seed=0, size=0):
        self.settings = settings
        self.data_path = os.path.join(path, dataset_type)

        files = list(Path(self.data_path).glob('*.*'))
        self.items = [str(file_path) for file_path in files]
        default_rng(seed).shuffle(self.items)
        if size != 0:
            self.items = self.items[:size]
        self.homography_config = HomographyConfig()
        self.do_augmentation = True
        self.transforms = dataset_transforms()

    def __getitem__(self, index):
        file_name = self.items[index]
        try:
            item_data = np.load(file_name)
        except Exception as e:
            print(f'CocoDataset failed to load file {file_name}')
            raise

        if self.do_augmentation:
            image = item_data['image'] * 255
            image = image.astype(np.uint8)
            image = image.transpose([1, 2, 0])
            augmented_image = self.transforms(image=image)
            image = augmented_image['image'].float() / 255.
        else:
            image = torch.from_numpy(item_data['image'])

        points = torch.from_numpy(item_data['points'])

        # take only coordinates
        points = points[:2, :]
        points = torch.transpose(points, 1, 0)
        # swap x and y columns
        points[:, [0, 1]] = points[:, [1, 0]]

        warped_image, warped_points, valid_mask, homography = homographic_augmentation(image.unsqueeze(0), points,
                                                                                       self.homography_config)

        img_h, img_w = image.shape[-2:]
        point_labels = make_points_labels(points.numpy(), img_h, img_w, self.settings.cell)
        warped_point_labels = make_points_labels(warped_points.numpy(), img_h, img_w, self.settings.cell)
        valid_mask = scale_valid_map(valid_mask, img_h, img_w, self.settings.cell)
        
        return image, torch.from_numpy(point_labels), warped_image.squeeze(dim=0), torch.from_numpy(
            warped_point_labels), valid_mask, homography

    def __len__(self):
        return len(self.items)
