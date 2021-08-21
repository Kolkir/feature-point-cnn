import albumentations as A
from albumentations.pytorch import ToTensorV2


def dataset_transforms():
    p = 1./3.
    transforms = A.Compose([
        A.RandomBrightnessContrast(p=p),
        A.OneOf([
            A.MotionBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
            A.Blur(blur_limit=3),
        ], p=p),
        A.OneOf([
            A.MultiplicativeNoise(),
            A.GaussNoise(),
        ], p=p),
        ToTensorV2()
    ])

    return transforms
