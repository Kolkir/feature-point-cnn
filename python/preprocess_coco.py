import os

import torch
import torchvision
import numpy as np
from inferencewrapper import InferenceWrapper
from homographies import HomographyConfig
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F


class CocoPreprocessDataset(Dataset):
    def __init__(self, path, settings, dataset_type, seed=0):
        self.settings = settings
        self.data_path = os.path.join(path, dataset_type)

        files = list(Path(self.data_path).glob('*.*'))
        self.items = [str(file_path) for file_path in files]

    def __getitem__(self, index):
        image_path = self.items[index]
        img = torchvision.io.image.read_image(str(image_path), mode=torchvision.io.image.ImageReadMode.GRAY)
        img = img.float().div(255)
        # ratio preserving resize
        _, img_h, img_w = img.shape
        scale_h = self.settings.train_image_size[0] / img_h
        scale_w = self.settings.train_image_size[1] / img_w
        scale_max = max(scale_h, scale_w)
        new_size = [int(img_h * scale_max), int(img_w * scale_max)]
        img = F.resize(img, new_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        img = F.center_crop(img, self.settings.train_image_size)

        return img, str(image_path)

    def __len__(self):
        return len(self.items)


def preprocess_coco(coco_path, magic_point_path, settings):
    print('Pre-process training COCO images:\n')

    print('Loading pre-trained Magic network...')
    net_wrapper = InferenceWrapper(weights_path=magic_point_path, settings=settings)
    print('Successfully loaded pre-trained network.')

    batch_size = 4
    num_workers = 4

    train_dataset = CocoPreprocessDataset(coco_path, settings, 'train2014')
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_dataset = CocoPreprocessDataset(coco_path, settings, 'test2014')
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    homo_config = HomographyConfig()
    homo_config.init_for_preprocess()

    preprocess_coco_folder(train_data_loader, net_wrapper, homo_config, Path(coco_path, 'train'))
    preprocess_coco_folder(test_data_loader, net_wrapper, homo_config, Path(coco_path, 'test'))


def preprocess_coco_folder(data_loader, net_wrapper, homo_config, output_path):
    output_path.mkdir(parents=True, exist_ok=True)
    for image_batch, image_path in tqdm(data_loader):
        points = net_wrapper.run_with_homography_adaptation(image_batch, homo_config)

        images = torch.unbind(image_batch)

        for i, image in enumerate(images):
            filename = Path(image_path[i]).stem
            filename = Path(output_path, f'{filename}.npz')
            np.savez_compressed(filename, image=image.numpy(), points=points[i])
