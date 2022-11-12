import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import cv2
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import get_transforms

noise_img = ['2017_KAG_2LB_000912.png', '2019_KBG_3LB_000073.png', '2019_KDG_2LB_000871.png', '2019_WSN_2LB_000047.png', '2019_JNG_1LB_000005.png', '2019_KSG_2LB_000374.png', '2019_SMG_1LB_000027.png']

noise_2n3 = ['2015_DMG_2LB_000436.png', '2016_YDP_JJG_000151.png', '2017_KSG_SAG_000148.png', '2017_YDP_2LB_000503.png', '2018_KAG_SAG_000041.png', '2018_KSG_JJG_000007.png', '2018_SMG_3LB_000185.png', '2018_SPG_3LB_000326.png', '2019_JNG_SAG_000234.png', '2019_JRG_SAG_000081.png', '2019_KSG_2LB_000544.png', '2019_KSG_SAG_000123.png', '2019_MPG_3LB_000229.png']

noise_1n2 = ['2016_YDP_JJG_000075.png', '2018_SCG_2LB_000288.png', '2018_SCG_2LB_000292.png', '2019_JNG_1LB_000129.png', '2019_KSG_2LB_000589.png', '2019_KSG_KNI_000461.png']

def get_loaders(args):
    data_path = args.dataset_path
    dataset = CDDataset(os.path.join(data_path, 'train'), img_size=(args.patch_size, args.patch_size), transform = get_transforms(True))
#     val_dataset = CDDataset(os.path.join(data_path, 'val'), transform = get_transforms(False))
    dataset_size = len(dataset)
    
    validation_size = int(dataset_size * args.val_ratio)
    train_size = int(dataset_size - validation_size)
    train_dataset, val_dataset = random_split(dataset, [train_size, validation_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )
    
    return train_loader, val_loader

def get_test_loader(args):
    data_path = args.dataset_path
    test_dataset = CDDataset(os.path.join(data_path, 'test'), img_size=(args.patch_size, args.patch_size), transform = get_transforms(False))  #, transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )
    
    return test_loader

#util
def is_image_file(filename):
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif', 'tiff'
    ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname) and (not fname in noise_img) and (not fname in noise_2n3) and (not fname in noise_1n2):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

class CDDataset(Dataset):
    def __init__(self, data_path, img_size=(256, 256), transform=None):
        self.data_path = data_path
        self.transform = transform
        self.img_size = img_size
        folder_A = 'x'
#         folder_B = ''
        folder_L = 'y'
        self.A_paths = sorted(make_dataset(os.path.join(self.data_path, folder_A)))
#         self.B_paths = sorted(make_dataset(os.path.join(self.data_path, folder_B)))
        self.L_paths = sorted(make_dataset(os.path.join(self.data_path, folder_L)))
        

    def __getitem__(self, index):
        A_path = self.A_paths[index]
#         B_path = self.B_paths[index]
        L_path = self.L_paths[index]
        
        # split img
        img = Image.open(A_path)
        img = np.asarray(img)
        img_shape = img.shape
        img1 = Image.fromarray(img[:, :img_shape[1]//2, :])
        img2 = Image.fromarray(img[:, img_shape[1]//2:, :])
        img1 = np.asarray(img1)
        img2 = np.asarray(img2)
        img1 = cv2.resize(img1, dsize = (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_CUBIC)
        img2 = cv2.resize(img2, dsize = (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_CUBIC)
        
#         B_img = Image.open(B_path)

        # Label
        label_img = Image.open(L_path)
        label_img = np.asarray(label_img)
        label_img_shape = label_img.shape
        label1 = Image.fromarray(label_img[:, :label_img_shape[1]//2])
        label2 = Image.fromarray(label_img[:, label_img_shape[1]//2:])
        label1 = label1.crop([5,5,(label_img_shape[1]//2)-5,label_img_shape[0]-5])
        label1 = np.asarray(label1,dtype=np.int32)
        label2 = label2.crop([5,5,(label_img_shape[1]//2)-5,label_img_shape[0]-5])
        label2 = np.asarray(label2,dtype=np.int32)
        label1 = cv2.resize(label1, dsize = (754, 754), interpolation=cv2.INTER_NEAREST)
        label2 = cv2.resize(label2, dsize = (754, 754), interpolation=cv2.INTER_NEAREST)
        label1 = Image.fromarray(label1)
        label2 = Image.fromarray(label2)
        label1 = label1.crop([5,5,749,749])
        label1 = np.asarray(label1,dtype=np.int32)
        label2 = label2.crop([5,5,749,749])
        label2 = np.asarray(label2,dtype=np.int32)
        label = label1 + label2
        label = cv2.resize(label, dsize = (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        label = Image.fromarray(label)
        sample = {'image': (img1, img2), 'label': label}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample['image'][0], sample['image'][1], sample['label']

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
    
