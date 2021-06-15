import torchvision
import numpy as np
from inferencewrapper import InferenceWrapper
from pathlib import Path
from tqdm import tqdm


def preprocess_coco(coco_path, magic_point_path, settings):
    print('Pre-process training COCO images:\n')

    print('Loading pre-trained Magic network...')
    net = InferenceWrapper(weights_path=magic_point_path, settings=settings)
    print('Successfully loaded pre-trained network.')

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(settings.train_image_size),
    ])

    preprocess_coco_folder(coco_path, 'train2014', net, transforms)
    preprocess_coco_folder(coco_path, 'test2014', net, transforms)


def preprocess_coco_folder(coco_path, folder, net, transforms):
    base_path = Path(coco_path, folder)
    output_path = Path(coco_path, f'{folder}_output')
    output_path.mkdir(parents=True, exist_ok=True)
    image_paths = list(base_path.iterdir())
    images_num = len(image_paths)
    for i in tqdm(range(images_num)):
        image_path = image_paths[i]
        img = torchvision.io.image.read_image(str(image_path), mode=torchvision.io.image.ImageReadMode.GRAY)
        img = transforms(img)
        img = img.float().div(255)
        points, _ = net.run(img)
        filename = Path(image_path).stem
        filename = Path(output_path, f'{filename}.npz')
        np.savez_compressed(filename, image=img.numpy(), points=points)
