import itertools
import os
import random
import sys
from math import trunc

import cv2
import numpy as np
from albumentations import (CLAHE, Blur, Compose, Flip, GaussNoise,
                            GridDistortion, HorizontalFlip, HueSaturationValue,
                            IAAAdditiveGaussianNoise, IAAEmboss,
                            IAAPerspective, IAAPiecewiseAffine, IAASharpen,
                            MedianBlur, MotionBlur, OneOf, OpticalDistortion,
                            PadIfNeeded, RandomBrightness,
                            RandomBrightnessContrast, RandomContrast,
                            RandomCrop, RandomRotate90, RandomSizedCrop,
                            Resize, ShiftScaleRotate, Transpose)

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNEL = 3


def val_aug(p=1):

    return Compose([
        # Resize(IMAGE_WIDTH,IMAGE_WIDTH,p=1),
        # PadIfNeeded(min_height=IMAGE_HEIGHT, min_width=IMAGE_HEIGHT, p=1),
        # RandomCrop(height=IMAGE_HEIGHT, width=IMAGE_HEIGHT, p=1),
        # Resize(IMAGE_WIDTH,IMAGE_WIDTH,p=1),
        # RandomRotate90(p=0.5),
        # Flip(p=0.5),
        # Transpose(p=0.5),
        # RandomSizedCrop(min_max_height=(200,200),height=IMAGE_HEIGHT,width=IMAGE_HEIGHT,p=0.5),
    ], p=p)


def train_aug(p=1):

    return Compose([
        # RandomRotate90(),
        # Flip(),
        # Transpose(),
        # OneOf([
        #     IAAAdditiveGaussianNoise(),
        #     GaussNoise(),
        # ], p=0.2),
        # OneOf([
        #     MotionBlur(p=0.2),
        #     MedianBlur(blur_limit=3, p=0.1),
        #     Blur(blur_limit=3, p=0.1),
        # ], p=0.2),
        # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        # OneOf([
        #     OpticalDistortion(p=0.3),
        #     GridDistortion(p=0.1),
        #     IAAPiecewiseAffine(p=0.3),
        # ], p=0.2),
        # OneOf([
        #     CLAHE(clip_limit=2),
        #     IAASharpen(),
        #     IAAEmboss(),
        #     RandomBrightnessContrast(),
        # ], p=0.3),
        # HueSaturationValue(p=0.3),
    ], p=p, additional_targets={"image2": "image"})


def image_to_channel(img1, img2):

    image = np.zeros((IMAGE_WIDTH, IMAGE_WIDTH, 6))
    image[:, :, :3] = img1
    image[:, :, 3:] = img2

    return image


def get_image(path):

    try:

        img = cv2.imread(path, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_HEIGHT, IMAGE_HEIGHT))

    except:

        print(path)

    return img


def convert_(img, mask):

    img = np.float32(img)
    img = img / 255.0
    mask = np.float32(mask)

    return img, np.expand_dims(mask, axis=-1)


def get_mask(path):

    try:

        mask = cv2.imread(path, 0)
        mask = cv2.resize(mask, (IMAGE_HEIGHT, IMAGE_WIDTH))
        mask[mask != 0] = 1

    except:

        print("error", path)

    return mask


def image_generator(input1, input2, masks, batch_size=8, one_hot_label=False, data_aug=False, isTrain=True):

    input1_zipped = itertools.cycle(zip(input1))
    input2_zipped = itertools.cycle(zip(input2))
    masks_zipped = itertools.cycle(zip(masks))

    while True:

        batch_input = []
        batch_output = []
        batch_input = []
        batch_input1 = []
        batch_input2 = []
        batch_output = []

        for _ in range(batch_size):

            input1_path = next(input1_zipped)[0]
            input2_path = next(input2_zipped)[0]
            mask_path = next(masks_zipped)[0]

            assert os.path.basename(mask_path) == os.path.basename(
                input1_path) == os.path.basename(input2_path), "Images name shoud be same"

            if not (os.path.exists(mask_path) or os.path.exists(input1_path)):

                print(input1_path, mask_path)

            input1 = get_image(input1_path)
            input2 = get_image(input2_path)

            output = get_mask(mask_path)

            if isTrain:

                augmentation = train_aug(p=1)

                if bool(random.getrandbits(1)):

                    input1, input2 = input2, input1

                data = {"image": input1, "image2": input2, "mask": output}
                augmented = augmentation(**data)

                input1, input2, output = augmented["image"], augmented["image2"], augmented["mask"]

            else:

                augmentation = val_aug(p=1)
                data = {"image": input1, "image2": input2, "mask": output}
                augmented = augmentation(**data)

                input1, input2, output = augmented["image"], augmented["image2"], augmented["mask"]

            input = image_to_channel(input1, input2)
            input, output = convert_(input, output)
            image1 = input[:, :, :3]
            image2 = input[:, :, 3:]
            batch_input.append(input)
            batch_input1.append(image1)
            batch_input2.append(image2)
            batch_output.append(output.astype(int))

        batch_x = np.array(batch_input)
        batch_x1 = np.array(batch_input1)
        batch_x2 = np.array(batch_input2)
        batch_y = np.array(batch_output)

        yield(batch_x, batch_y)