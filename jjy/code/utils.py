from loss import FocalLoss, dice_loss
import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms

# Train and Valid metrics
def initialize_metrics():
    """Generates a dictionary of metrics with metrics as keys
       and empty lists as values
    Returns
    -------
    dict
        a dictionary of metrics
    """
    metrics = {
        'cd_losses': [],
        'cd_corrects': [],
        'cd_precisions': [],
        'cd_recalls': [],
        'cd_f1scores': [],
        'learning_rate': [],
    }

    return metrics

def get_mean_metrics(metric_dict):
    """takes a dictionary of lists for metrics and returns dict of mean values
    Parameters
    ----------
    metric_dict : dict
        A dictionary of metrics
    Returns
    -------
    dict
        dict of floats that reflect mean metric value
    """
    return {k: np.mean(v) for k, v in metric_dict.items()}


def set_metrics(metric_dict, cd_loss, cd_corrects, cd_report, lr):
    """Updates metric dict with batch metrics
    Parameters
    ----------
    metric_dict : dict
        dict of metrics
    cd_loss : dict(?)
        loss value
    cd_corrects : dict(?)
        number of correct results (to generate accuracy
    cd_report : list
        precision, recall, f1 values
    Returns
    -------
    dict
        dict of  updated metrics
    """
    metric_dict['cd_losses'].append(cd_loss.item())
    metric_dict['cd_corrects'].append(cd_corrects.item())
    metric_dict['cd_precisions'].append(cd_report[0])
    metric_dict['cd_recalls'].append(cd_report[1])
    metric_dict['cd_f1scores'].append(cd_report[2])
    metric_dict['learning_rate'].append(lr)

    return metric_dict

def set_test_metrics(metric_dict, cd_corrects, cd_report):

    metric_dict['cd_corrects'].append(cd_corrects.item())
    metric_dict['cd_precisions'].append(cd_report[0])
    metric_dict['cd_recalls'].append(cd_report[1])
    metric_dict['cd_f1scores'].append(cd_report[2])

    return metric_dict

# loss Function
def hybrid_loss(predictions, target):
    """Calculating the loss"""
    loss = 0

    # gamma=0, alpha=None --> CE
    
    focal = FocalLoss(gamma=0, alpha=None)

    for prediction in predictions:
        bce = focal(prediction, target)
        dice = dice_loss(prediction, target)
        loss += bce + dice

    return loss

# transfrom util
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        img1 = np.array(img1).astype(np.float32).transpose((2, 0, 1))
        img2 = np.array(img2).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32) / 255.0

        img1 = torch.from_numpy(img1).float()
        img2 = torch.from_numpy(img2).float()
        mask = torch.from_numpy(mask).float()

        return {'image': (img1, img2),
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': (img1, img2),
                'label': mask}

class RandomVerticalFlip(object):
    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < 0.5:
            img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
            img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return {'image': (img1, img2),
                'label': mask}

class RandomFixRotate(object):
    def __init__(self):
        self.degree = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < 0.75:
            rotate_degree = random.choice(self.degree)
            img1 = img1.transpose(rotate_degree)
            img2 = img2.transpose(rotate_degree)
            mask = mask.transpose(rotate_degree)

        return {'image': (img1, img2),
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img1 = img1.rotate(rotate_degree, Image.BILINEAR)
        img2 = img2.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': (img1, img2),
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < 0.5:
            img1 = img1.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            img2 = img2.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': (img1, img2),
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']

        assert img1.size == mask.size and img2.size == mask.size

        img1 = img1.resize(self.size, Image.BILINEAR)
        img2 = img2.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': (img1, img2),
                'label': mask}
    
# transforms setting
def get_transforms(istrain=False):
    if istrain:
        transform = transforms.Compose([
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                RandomFixRotate(),
                # RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
                # RandomGaussianBlur(),
                # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensor()])
    else:
        transform = transforms.Compose([
                # RandomHorizontalFlip(),
                # RandomVerticalFlip(),
                # RandomFixRotate(),
                # RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
                # RandomGaussianBlur(),
                # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensor()])
    return transform