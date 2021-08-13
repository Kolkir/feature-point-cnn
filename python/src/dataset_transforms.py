import albumentations as A
from albumentations.pytorch import ToTensorV2


def dataset_transforms():
    p = 1./3.
    transforms = A.Compose([
        A.RandomBrightnessContrast(p=p),
        A.OneOf([
            A.MotionBlur(),
            A.MedianBlur(),
            A.Blur(),
        ], p=p),
        A.OneOf([
            A.MultiplicativeNoise(),
            A.GaussNoise(),
        ], p=p),
        ToTensorV2()
    ])

    return transforms
