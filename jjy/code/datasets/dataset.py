import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import get_transforms

def get_loaders(args):
    data_path = args.dataset_path
    train_dataset = CDDataset(os.path.join(data_path, 'train'), transform = get_transforms(True))  #, transform)
    val_dataset = CDDataset(os.path.join(data_path, 'val'), transform = get_transforms(False))
    
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
    test_dataset = CDDataset(os.path.join(data_path, 'test'), transform = get_transforms(False))  #, transform)
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
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

class CDDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        folder_A = 'A'
        folder_B = 'B'
        folder_L = 'OUT'
        self.A_paths = sorted(make_dataset(os.path.join(self.data_path, folder_A)))
        self.B_paths = sorted(make_dataset(os.path.join(self.data_path, folder_B)))
        self.L_paths = sorted(make_dataset(os.path.join(self.data_path, folder_B)))
        

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        L_path = self.L_paths[index]
        
        A_img = Image.open(A_path)
        B_img = Image.open(B_path)
        label = Image.open(L_path).convert("L")
        sample = {'image': (A_img, B_img), 'label': label}
        
        
        if self.transform:
            sample = self.transform(sample)
        
#         A = np.array(A_img).astype(np.float32).transpose((2, 0, 1))
#         B = np.array(B_img).astype(np.float32).transpose((2, 0, 1))
        
#         L_path = self.L_paths[index]
        
        # 1번 방법
#         L_s = Image.open(L_path)
#         mask = np.array(L_s).astype(np.float32) / 255.0
#         mask = np.array(mask).astype(np.float32).transpose((2, 0, 1))

#         A = torch.from_numpy(A).float()
#         B = torch.from_numpy(B).float()
        
#         L = torch.from_numpy(mask).float()
        # 2번 방법
#         tmp = np.array(Image.open(L_path), dtype=np.uint32)/255
#         L_img = Image.fromarray(tmp)
#         L_s = L_img.float()
#         L_s = F.interpolate(L_s, size=torch.Size([A_img.shape[2], A_img.shape[3]]),mode='nearest')
#         L_s[self.L_s == 1] = -1  # change
#         L_s[self.L_s == 0] = 1  # no change
        
#         if transform:
#             pass
        
        return sample['image'][0], sample['image'][1], sample['label']

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
    
