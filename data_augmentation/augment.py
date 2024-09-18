import random
import os
import cv2
from moire import Moire
import random
from tqdm import tqdm

root = "/data/datasets/thainq/sonnt373/dev/FAS/data/Data_FAS_v0"
file_path = os.path.join(root, "train.txt")
count_live = 0

label_save = os.path.join(root, "train_aug.txt")

moire = Moire()
with open(file_path, "r") as file:
    lines = file.readlines()
    
    random.shuffle(lines)
    with open(label_save, '+a') as file_save:
        for line in tqdm(lines):
            items = line.strip().split()
            img_path = os.path.join(root, items[0])
            label = items[1]
            if random.uniform(0,1) < 0.1:
                image_org = cv2.imread(img_path)
                # print('=== ', image.shape, label)
                # cv2.imwrite(f"face_{label}.jpg", image)
                image_aug = moire(image_org)
                path_output = items[0].replace("celeb_rose", "celeb_rose_aug")

                print("path_output: ", path_output)
                file_save.write(f"{path_output} {1}\n")
                cv2.imwrite(os.path.join(root, path_output), image_aug)
                # break


    # print("count_live :: ", count_live)
    # print("count_spoof :: ", len(lines) -  count_live)

    print("count_all :: ", len(lines))
