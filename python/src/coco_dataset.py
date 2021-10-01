import torch
from torch.utils.data import Dataset
import os
from pathlib import Path
from numpy.random import default_rng

from src.dataset_transforms import dataset_transforms
from src.dataset_utils import read_dataset_item
from src.homographies import homographic_augmentation, HomographyConfig
from src.netutils import make_points_labels, scale_valid_map


class CocoDataset(Dataset):
    def __init__(self, path, settings, dataset_type, do_augmentation=True, seed=0, size=0):
        self.settings = settings
        self.data_path = os.path.join(path, dataset_type)

        files = list(Path(self.data_path).glob('*.*'))
        self.items = [str(file_path) for file_path in files]
        default_rng(seed).shuffle(self.items)
        if size != 0:
            self.items = self.items[:size]
        self.homography_config = HomographyConfig()
        self.do_augmentation = do_augmentation
        self.transforms = dataset_transforms()

    def __getitem__(self, index):
        file_name = self.items[index]
        image, points = read_dataset_item(file_name, self.transforms if self.do_augmentation else None)

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
