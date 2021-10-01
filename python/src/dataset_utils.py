import numpy as np
import torch


def read_dataset_item(file_name, transforms=None):
    try:
        item_data = np.load(file_name)
    except Exception as e:
        print(f'Dataset failed to load file {file_name}')
        raise

    image = item_data['image']
    assert (len(image.shape) >= 2)
    if len(image.shape) < 3:
        # add missed channel dimension
        image = np.expand_dims(image, axis=0)

    # make image color
    if image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.

    if transforms is not None:
        image = image * 255
        image = image.astype(np.uint8)
        image = image.transpose([1, 2, 0])
        augmented_image = transforms(image=image)
        image = augmented_image['image'].float() / 255.
    else:
        image = torch.from_numpy(image)

    points = torch.from_numpy(item_data['points'])

    # take only coordinates
    points = points[:2, :]
    points = torch.transpose(points, 1, 0)
    # swap x and y columns
    points[:, [0, 1]] = points[:, [1, 0]]

    return image, points
