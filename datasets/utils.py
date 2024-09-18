import os
import sys
import os.path
import torch
from datasets.CvprDataset_P1 import CvprDataset_P1
from datasets.CvprDataset_P21 import CvprDataset_P21
from datasets.CvprDataset_P22 import CvprDataset_P22
from cv2_transform import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_dataset(data_root, data_list, transforms1, transforms2, is_train, return_path=False):
    # if args.protocol == 'p1':
    _dataset = CvprDataset_P1(
        basedir = data_root,
        data_list = data_list,
        transforms1 = transforms1,
        transforms2 = transforms2,
        is_train = is_train,
        return_path = return_path)

    return _dataset


def get_train_dataset(data_root, data_list, input_size, return_path=False):
    transforms1 = None
    transforms2 = None

    # todo: update augment blur
    transforms1 = transforms.Compose([
        # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.7)),
        # transforms.Resize(size=(240, 240)),
        # transforms.RandomCrop(input_size),
        transforms.RandomResizedCrop(size=input_size, scale=(0.9, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
        transforms.ColorTrans(mode=0), # BGR to RGB
    ])

    transforms2 = A.Compose([
        # A.Blur(blur_limit=3, p=0.2),
        # A.MotionBlur(blur_limit=5, p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # RGB [0,255] input, RGB normalize output
        ToTensorV2(),
    ])
    _dataset = get_dataset(data_root, data_list, transforms1=transforms1, transforms2=transforms2, is_train=True, return_path=return_path)
    return _dataset


def get_val_dataset(data_root, data_list, input_size, return_path=False):
    transforms1 = None
    transforms2 = None

    transforms1 = transforms.Compose([
        transforms.Resize(size=(input_size, input_size)),
        transforms.ColorTrans(mode=0), # BGR to RGB
    ])
    transforms2 = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # RGB [0,255] input, RGB normalize output
        # A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)), # RGB [0,255] input, RGB normalize output
        ToTensorV2(),
    ])

    _dataset = get_dataset(data_root, data_list, transforms1=transforms1, transforms2=transforms2, is_train=False, return_path=return_path)
    return _dataset


