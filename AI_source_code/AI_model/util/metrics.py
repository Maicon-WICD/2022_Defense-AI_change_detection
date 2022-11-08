import glob
import os

import cv2
import numpy as np
from shapely.geometry import *

from util import custom_data_generator as data_util

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNEL = 3


def calculate_IoU_Per_Epoch_1(model, val_images1_list, val_images2_list, val_labels_list, checkpoint_path, epoch_number, write_images=True, tta=False):

    save_path = checkpoint_path + "Epoch_" + str(epoch_number) + "/"

    if not os.path.exists(save_path):

        os.makedirs(save_path)

    IU_BG_TP, IU_BG_FN, IU_BG_FP, IU_BG_TN, IU_BD_TP, IU_BD_FN, IU_BD_FP, IU_BD_TN = [
    ], [], [], [], [], [], [], []

    for j in range(len(val_images1_list)):

        gt = data_util.get_mask(val_labels_list[j]).astype(int)

        image1, gt = data_util.convert_(
            data_util.get_image(val_images1_list[j]), gt)
        image2, _ = data_util.convert_(
            data_util.get_image(val_images2_list[j]), gt)
        augmentation = data_util.val_aug(p=1)
        data = {"image": image1, "image2": image2, "mask": gt}
        augmented = augmentation(**data)
        image1, image2, gt = augmented["image"], augmented["image2"], augmented["mask"]
        input_image = data_util.image_to_channel(image1, image2)

        if tta:

            tta_model = tta_wrapper.TTA_ModelWrapper(model)
            pred = tta_model.predict(input_image)

        else:
            
            pred = model.predict(np.expand_dims(
                input_image, axis=0), batch_size=None, verbose=0, steps=None)

        pred[pred > 0.3] = 1
        pred = np.round(pred[0, :, :, 0]).astype(int)
        gt = np.round(gt[:, :, 0]).astype(int)
        classes = np.array([0, 1])

        for ii in classes:

            TP, FN, FP, TN = IoU(pred, gt, ii)

            if ii == 0:

                IU_BG_TP.append(TP)
                IU_BG_FN.append(FN)
                IU_BG_FP.append(FP)
                IU_BG_TN.append(TN)

            elif ii == 1:

                IU_BD_TP.append(TP)
                IU_BD_FN.append(FN)
                IU_BD_FP.append(FP)
                IU_BD_TN.append(TN)

        gt_pred = np.concatenate(
            (np.stack((pred,)*3, axis=-1), np.stack((gt,)*3, axis=-1)), axis=1)
        gt_pred[gt_pred != 0] = 255

        cv2.imwrite(save_path+os.path.basename(val_labels_list[j]), np.concatenate(
            (input_image[:, :, :3]*255, input_image[:, :, 3:]*255, gt_pred), axis=1))

    BG_IU = divided_IoU(IU_BG_TP, IU_BG_FN, IU_BG_FP)
    BD_IU = divided_IoU(IU_BD_TP, IU_BD_FN, IU_BD_FP)
    BG_P = divided_PixelAcc(IU_BG_TP, IU_BG_FN)
    BD_P = divided_PixelAcc(IU_BD_TP, IU_BD_FN)
    precision_ = precision(IU_BD_TP, IU_BD_FP)
    recall_ = recall(IU_BD_TP, IU_BD_FN)
    f_score_ = f_score(precision_, recall_, 1)

    return BG_IU, BD_IU, BG_P, BD_P, precision_, recall_, f_score_


def IoU(pred, valid, cl):

    tp = np.count_nonzero(np.logical_and(pred == cl, valid == cl))
    fn = np.count_nonzero(np.logical_and(pred != cl, valid == cl))
    fp = np.count_nonzero(np.logical_and(pred == cl, valid != cl))
    tn = np.count_nonzero(np.logical_and(pred != cl, valid != cl))

    return tp, fn, fp, tn


def divided_IoU(tp, fn, fp):

    try:

        return float(sum(tp)) / (sum(tp) + sum(fn) + sum(fp))

    except ZeroDivisionError:

        return 0


def divided_PixelAcc(tp, fn):

    try:

        return float(sum(tp)) / (sum(tp) + sum(fn))

    except ZeroDivisionError:

        return 0


def precision(tp, fp):

    try:
        
        return float(sum(tp)) / (sum(tp) + sum(fp))

    except ZeroDivisionError:

        return 0


def recall(tp, fn):

    try:

        return float(sum(tp)) / (sum(tp) + sum(fn))

    except ZeroDivisionError:

        return 0


def false_positive_rate(fp, tn):

    try:

        return float(fp / (fp + tn))
        
    except ZeroDivisionError:

        return 0


def false_negative_rate(fn, tp):

    try:

        return float(fn / (fn + tp))
        
    except ZeroDivisionError:

        return 0


'''
F1-score is the harmonic mean of precision and
recall, so it is influenced by precision and recall with equal
strength. We also use F2-score, which weighs recall more
importantly than precision with beta = 2. Therefore, F2-score
gets a larger penalty by miss alarms than by false alarms. In
a surveillance system, false alarms can be double checked by
personnel while miss alarms do not have such opportunities.
Accordingly, we decide that F2-score is more suitable as an
assessment tool for change detection techniques
'''


def f_score(precision, recall, beta):

    try:

        return ((1 + beta*beta) * precision * recall) / ((beta*beta*precision) + recall)

    except ZeroDivisionError:

        return 0


def get_polygons(img):

    contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []

    '''append polygons and contours of each polygons to list'''

    for i in range(len(contours)):

        cnt = contours[i]
        perimeter = cv2.arcLength(cnt, True)
        epsilon = float(0.007) * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx[:, 0, :]) > 3:

            poly = Polygon(approx[:, 0, :]).buffer(0)

            if poly.area > 15:

                polygons.append(poly)

    return polygons


def polygon_calculate(prediction, label):

    prediction_polygons = get_polygons(prediction)
    label_polygons = get_polygons(label)

    TP = 0
    FN = 0
    FP = 0

    for poly in label_polygons:

        i = 0

        for poly2 in prediction_polygons:

            intersection = poly.intersection(poly2).area

            if intersection:

                i += 1

                break

        if i == 1:

            TP += 1

        else:

            FN += 1

    FP = len(prediction_polygons) - TP

    return TP, FN, FP