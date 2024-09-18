from skimage.transform import PiecewiseAffineTransform, warp
import numpy as np
import cv2
import random
import albumentations as A
from albumentations import Compose, ChannelShuffle, RGBShift, HueSaturationValue, RandomBrightnessContrast, Downscale, Sharpen, LongestMaxSize, Resize, PadIfNeeded, CenterCrop, Affine
import joblib
import os
from tqdm import tqdm


def random_center_crop(image, ratio):

    h, w, _ = image.shape
    crop_size = int(ratio * max(h, w))
    crop_h, crop_w = crop_size, crop_size

    center_y, center_x = h // 2, w // 2
    start_x = max(0, center_x - crop_w // 2)
    start_y = max(0, center_y - crop_h // 2)

    cropped_image = image[start_y:start_y + crop_h, start_x:start_x + crop_w]

    return cropped_image

def augment_image(image_path, mask_path, folder_back="./back_grounds"):
    color_transform = A.Compose([
        ChannelShuffle(p=0.03),
        HueSaturationValue(hue_shift_limit=9, sat_shift_limit=9, val_shift_limit=20, always_apply=False, p=0.5),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, always_apply=False, p=1),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
        Downscale(scale_min=0.4, scale_max=0.65, interpolation=0, always_apply=False, p=0.4),
        Sharpen(alpha=(0.2, 0.5), lightness=(0.2, 1.0), always_apply=False, p=1),
    ])
    only_mask = True
    if random.uniform(0, 1) > 0.5:
        only_mask = False

    # random background_image
    background_image = random.choice(os.listdir(folder_back))
    background_image_path = os.path.join(folder_back, background_image)
    image_backgound = cv2.imread(background_image_path)

    image = cv2.imread(image_path)
    augmented = color_transform(image=image)
    image = augmented['image']
    background = cv2.resize(image_backgound, (image.shape[1], image.shape[0]))

    mask = joblib.load(mask_path)
    mask = mask.astype(np.uint8)

    # where_res = np.where((mask >= 1))
    if only_mask == True:
        where_res = np.where((mask>=1) & (mask<=8) | (mask>=10) & (mask<=13)) # 1-6 10-13
    else:
        where_res = np.where((mask >= 1))


    bin_mask = np.zeros_like(mask, dtype='float32')
    bin_mask[where_res] = 1
    mask = bin_mask.astype(np.uint8)

    # scale massk
    scale_factor = random.uniform(0.75, 0.85)
    if only_mask == True:
        scale_factor = 1
    new_size = (int(mask.shape[1] * scale_factor), int(mask.shape[0] * scale_factor))
    small_mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_AREA)
    small_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    background_masked = background

    x_offset = (background.shape[1] - small_image.shape[1]) // 2
    y_offset = (background.shape[0] - small_image.shape[0]) // 2

    inverse_mask = cv2.bitwise_not(small_mask * 255)
    background_masked[y_offset:y_offset + small_mask.shape[0], x_offset:x_offset + small_mask.shape[1]] = cv2.bitwise_and(
        background[y_offset:y_offset + small_mask.shape[0], x_offset:x_offset + small_mask.shape[1]],
        background[y_offset:y_offset + small_mask.shape[0], x_offset:x_offset + small_mask.shape[1]],
        mask=inverse_mask
    )

    masked_image = cv2.bitwise_and(small_image, small_image, mask=small_mask)
    result = background_masked.copy()
    result[y_offset:y_offset + masked_image.shape[0], x_offset:x_offset + masked_image.shape[1]] = cv2.add(
        result[y_offset:y_offset + masked_image.shape[0], x_offset:x_offset + masked_image.shape[1]],
        masked_image
    )

    brightness = random.uniform(0.8, 1.2)
    result = cv2.convertScaleAbs(result, alpha=brightness)

    contrast = random.uniform(0.8, 1.2)
    result = cv2.convertScaleAbs(result, beta=128 * (1 - contrast), alpha=contrast)

    noise = np.random.normal(0, 5, result.shape).astype(np.int16)
    result = np.clip(result + noise, 0, 255).astype(np.uint8)

    # blur_kernel_size = random.choice([1, 3])
    # result = cv2.GaussianBlur(result, (blur_kernel_size, blur_kernel_size), 0)

    return result


root = "/data/datasets/thainq/sonnt373/dev/FAS/data/Data_FAS_v0"
file_path = os.path.join(root, "train.txt")
count_live = 0

label_save = os.path.join(root, "train_aug_back.txt")

with open(file_path, "r") as file:
    lines = file.readlines()
    
    random.shuffle(lines)
    with open(label_save, '+a') as file_save:
        for line in tqdm(lines):
            items = line.strip().split()
            img_path = os.path.join(root, items[0])
            mask_path = img_path.replace("celeb_rose", "train_live_mask")
            mask_path = mask_path.replace(".jpg", ".pkl")

            label = int(items[1])
            if os.path.exists(mask_path):
                if random.uniform(0,1) < 0.5 and label == 0:
                    image_aug = augment_image(img_path, mask_path)
                    image_aug = random_center_crop(image_aug, random.uniform(0.9, 1))
                    # cv2.imwrite("result.jpg", image_aug)
                    path_output = items[0].replace("celeb_rose", "celeb_rose_aug_back")
                    cv2.imwrite(os.path.join(root, path_output), image_aug)
                    file_save.write(f"{path_output} {1}\n")

