"""Augmentation pipelines (albumentations 2.0.8). Geometric transforms move the lesion mask with the image.

default_train  notebook 04 recipe: resize 224, flips, rotate +-20 deg, ColorJitter 0.2 (V0, V1, V2)
mix_train      Kozachok 2026a "mix" policy, Table A6, applied before the final resize (V3, V4)
eval_tf        resize only
"""
import albumentations as A
import cv2

SIZE = 224


def default_train(size=SIZE):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=1.0),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=1.0),
    ])


def mix_train(size=SIZE):
    """Table A6 of Kozachok 2026a, in order. Inside the photometric OneOf, each member's p is its
    relative weight; the two-operator Compose members keep each operator's own p."""
    cj = lambda p: A.ColorJitter(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=(-0.1, 0.1), p=p)
    pj = lambda p: A.PlanckianJitter(temperature_limit=(3000, 11000), p=p)
    hsv = lambda p: A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=30, p=p)
    he = lambda p: A.HEStain(method="preset", preset="macenko", intensity_scale_range=(0.9, 1.1),
                             intensity_shift_range=(-0.1, 0.1), p=p)
    return A.Compose([
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=(-0.1, 0.1), scale=(0.85, 1.15), rotate=(-360, 360), shear=0,
                 border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.4),
        A.Rotate(limit=360, crop_border=True, p=0.8),
        A.CLAHE(clip_limit=4, tile_grid_size=(8, 8), p=0.4),
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=5, p=1.0),
            A.GaussNoise(std_range=(0.02, 0.11), p=1.0),
        ], p=0.7),
        A.OneOf([
            A.OpticalDistortion(distort_limit=(-1.0, 1.0), p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=(-1.0, 1.0), p=1.0),
            A.ElasticTransform(alpha=3, p=1.0),
        ], p=0.7),
        A.OneOf([
            A.Compose([cj(0.5), pj(0.3)]),
            A.Compose([cj(0.5), hsv(0.3)]),
            A.Compose([he(0.2), pj(0.3)]),
            A.Compose([he(0.2), cj(0.5)]),
            A.Compose([he(0.2), hsv(0.3)]),
            pj(0.3),
            cj(0.5),
        ], p=0.7),
        A.ToGray(p=0.001),
        A.ChromaticAberration(primary_distortion_limit=(-0.1, 0.1), secondary_distortion_limit=(-0.1, 0.1),
                              mode="random", p=0.5),
        A.RandomGamma(gamma_limit=(40, 160), p=0.5),
        A.Resize(size, size),
    ])


def eval_tf(size=SIZE):
    return A.Compose([A.Resize(size, size)])
