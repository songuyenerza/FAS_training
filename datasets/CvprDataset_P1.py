import os
import torch
import cv2
import random
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from torchvision import transforms
from PIL import Image, ImageOps
from data_augmentation.moire import Moire
from data_augmentation.digital import digital_augment
import random


class CvprDataset_P1(Dataset):
    def __init__(self, basedir, data_list, transforms1=None, transforms2=None, is_train=True, return_path=False):
        self.base_path = basedir
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.return_path = return_path
        self.items = []
        self.is_train = is_train

        # data_augmentation
        self.moire = Moire()

        # center crop
        self.bbox_delta = 250

        with open(data_list, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if is_train:
                    items = line.strip().split()
                    img_path = items[0]
                    label = items[1]
                    self.items.append((img_path, label))
                else:
                    items = line.strip().split()
                    img_path = items[0]
                    if len(items) == 1:
                        label = "1"
                    else:
                        label = items[1]
                    self.items.append((img_path, label))

    def __getitem__(self, idx):

        fpath = self.items[idx][0]
        image = cv2.imread(os.path.join(self.base_path, fpath))

        label = int(self.items[idx][1])

        if self.transforms1 is not None:
            image = self.transforms1(image)

        if self.transforms2 is not None:
            image = self.transforms2(image=image)
            image = image['image']

        # image_np = np.array(image)
        # image_np = np.transpose(image_np, (1, 2, 0)) 
        # image_np = (image_np * 255).astype(np.uint8)
        # cv2.imwrite("check.jpg", image_np)

        if self.return_path:
            return image, label, fpath
        else:
            return image, label

    def __len__(self):
        return len(self.items)