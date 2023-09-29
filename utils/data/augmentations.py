import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.data_utils.transforms import apply_gamma_correction

train_transforms = A.Compose(
    [
            A.CLAHE(),
            A.RandomRotate90(),
            A.Transpose(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
            A.Blur(blur_limit=3),
            A.OpticalDistortion(),
            A.GridDistortion(),
            A.HueSaturationValue(),
            A.Lambda(image=apply_gamma_correction),
            ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [   
        A.Lambda(image=apply_gamma_correction),
        ToTensorV2(),
    ]
)
