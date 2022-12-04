"""Datasets
"""

from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import albumentations as A

class SegDataset(Dataset):
    """Dataset for image segmentation

    Attributs:
        x_dirs(list): 이미지 경로
        y_dirs(list): 마스크 이미지 경로
        input_size(list, tuple): 이미지 크기(width, height)
        scaler(obj): 이미지 스케일러 함수
        logger(obj): 로거 객체
        verbose(bool): 세부 로깅 여부
    """   
    def __init__(self, paths, input_size, scaler, mode='train', logger=None, verbose=False):
        
        self.x_paths = paths
        self.y_paths = list(map(lambda x : x.replace('x', 'y'),self.x_paths))
        self.input_size = input_size
        self.scaler = scaler
        self.logger = logger
        self.verbose = verbose
        self.mode = mode


    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, id_: int):

        filename = os.path.basename(self.x_paths[id_])  # Get filename for logging
        x = cv2.imread(self.x_paths[id_], cv2.IMREAD_COLOR)
        orig_size = x.shape

        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        if self.mode in ['train']:
            # x = cv2.resize(x, self.input_size)
            x = self.scaler(x)

            y = cv2.imread(self.y_paths[id_], cv2.IMREAD_GRAYSCALE)
            # y = cv2.resize(y, self.input_size, interpolation=cv2.INTER_NEAREST)

            #divide image(left,right)
            left_x,left_y = x[:,:int(x.shape[1]/2)], y[:,:int(x.shape[1]/2)]
            right_x,right_y = x[:,int(x.shape[1]/2):], y[:,int(x.shape[1]/2):]
            # Augmentation add
            transform = A.Compose([
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf([
                    A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                    A.GridDistortion(p=0.5),
                    A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
                ], p=0.8),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20,always_apply=False, p=1),
                A.Resize(width = int(self.input_size[0]/2),height = self.input_size[1])],
                additional_targets={"right_image":"image","right_mask":"mask"})
            transformed = transform(image=left_x,mask=left_y,right_image=right_x,right_mask = right_y)
            left_x = transformed['image']
            left_y = transformed['mask']
            right_x = transformed['right_image']
            right_y = transformed['right_mask']

            x = cv2.hconcat([left_x,right_x])
            y = cv2.hconcat([left_y,right_y])
            x = np.transpose(x, (2, 0, 1))

            return x, y, filename

        elif self.mode in ['valid']:
            x = cv2.resize(x, self.input_size)
            x = self.scaler(x)
            x = np.transpose(x, (2, 0, 1))

            y = cv2.imread(self.y_paths[id_], cv2.IMREAD_GRAYSCALE)
            y = cv2.resize(y, self.input_size, interpolation=cv2.INTER_NEAREST)

            return x, y, filename

        elif self.mode in ['test']:
            x = cv2.resize(x, self.input_size)
            x = self.scaler(x)
            x = np.transpose(x, (2, 0, 1))
            return x, orig_size, filename

        else:
            assert False, f"Invalid mode : {self.mode}"