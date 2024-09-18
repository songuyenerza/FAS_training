from __future__ import print_function
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class DatasetMask(Dataset):
    def __init__(self, all_live_data, root_dir, transform=None):
        self.img_paths = all_live_data
        self.root_dir = root_dir
        self.transform = transform
        print("number is: ", len(self.img_paths))

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(os.path.join(self.root_dir, img_path))
        image = img.resize((512, 512), Image.BILINEAR)

        image = self.transform(image)

        return img_path, image

    def __len__(self):
        return len(self.img_paths)


