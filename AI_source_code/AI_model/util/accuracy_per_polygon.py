import numpy as np
from PIL import Image
import cv2, glob,os
import itertools
from shapely.geometry import *
import geopandas as gp
import pandas as pd
import argparse
import statistics

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
#         return float(tp / (tp + fp))
        return float(sum(tp)) / (sum(tp) + sum(fp))
    except ZeroDivisionError:
        return 0

def recall(tp, fn):
    try:
#         return float(tp / (tp + fn))
        return float(sum(tp)) / (sum(tp) + sum(fn))
    except ZeroDivisionError:
        return 0

def false_positive_rate(fp, tn):
    try:
        return float(fp / (fp + tn))
        #return float(sum(fp)) / (sum(fp) + sum(tn))
    except ZeroDivisionError:
        return 0

def false_negative_rate(fn, tp):
    try:
        return float(fn / (fn + tp))
        #return float(sum(fn)) / (sum(fn) + sum(tp))
    except ZeroDivisionError:
        return 0

def f_score(precision, recall, beta):
    try:
        return (2 * (precision * recall) )/ (precision + recall)
    except ZeroDivisionError:
        return 0

def f1_score(precision, recall, beta):
    try:
        return ((1 + beta*beta) * precision * recall) / ((beta*beta*precision) + recall)
    except ZeroDivisionError:
        return 0


def get_polygons(img):
    # _, contours, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []

    '''append polygons and contours of each polygons to list'''
    for i in range(len(contours)):
        cnt = contours[i]
        perimeter = cv2.arcLength(cnt, True)
        epsilon = float(0.007) * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx[:, 0, :]) > 3:
            poly = Polygon(approx[:, 0, :]).buffer(0)

            if poly.area > 300:
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
            if poly.contains(poly2):
                i+=1
                break
            elif poly2.contains(poly):
                i+=1
                break
            else:
                intersection = poly.intersection(poly2).area
                union = poly.union(poly2).area
                IOU = intersection / union
                if IOU > 0.3:
                    i += 1
                    break
        if i == 1:
            TP += 1
        else:
            FN += 1

    FP = len(prediction_polygons) - TP
    
    return TP, FN, FP, len(prediction_polygons), len(label_polygons)

def cal(pred_list,gt_list):
    IU_BG_TP, IU_BG_FN, IU_BG_FP, IU_BG_TN, IU_BD_TP, IU_BD_FN, IU_BD_FP, IU_BD_TN = [], [], [], [], [], [], [], []
    P_TP, P_FN, P_FP = [],[],[]
    gt_poly, pred_poly = 0,0
    for i in range(len(pred_list)):
        pred = pred_list[i]
        gt = gt_list[i]
        for ii in [0,1]:
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
            
        tp,fn,fp,npred,ngt = polygon_calculate(pred,gt)
        pred_poly +=npred
        gt_poly +=ngt
        P_TP.append(tp)
        P_FN.append(fn)
        P_FP.append(fp)

    BG_IU =   divided_IoU(IU_BG_TP, IU_BG_FN, IU_BG_FP)
    BD_IU = divided_IoU(IU_BD_TP, IU_BD_FN, IU_BD_FP)
    BG_P =   divided_PixelAcc(IU_BG_TP, IU_BG_FN)
    BD_P =   divided_PixelAcc(IU_BD_TP, IU_BD_FN)
    precision_ = precision(IU_BD_TP,IU_BD_FP)
    recall_ = recall(IU_BD_TP,IU_BD_FN)
    f_score_pixel  = f1_score(precision_, recall_,1)
    
    f_score_poly = f1_score(precision(P_TP,P_FP), recall(P_TP,P_FN),1)

    return P_TP, P_FP, P_FN, precision_, recall_, f_score_poly

### Change path to prediction binary map and groundtruth here
result_f = '/workspace/AI_model/checkpoints/UNET/Epoch_98/' # <set ur own path here>
path = glob.glob(result_f + '*.png')

pred_f = result_f + 'pred/' 
gt_f = result_f + 'gt/'

if not os.path.exists(pred_f):
	os.makedirs(pred_f)

if not os.path.exists(gt_f):
	os.makedirs(gt_f)

for imagename in path:
    image = cv2.imread(imagename)
    thirdImage = image[:, 512:768, :]
    fourthImage = image[:, 768:1024, :]
    cv2.imwrite(pred_f + imagename.split('/')[-1], thirdImage)
    cv2.imwrite(gt_f + imagename.split('/')[-1], fourthImage)

pred_files = glob.glob("%s/*.png"%pred_f)
gt_files = glob.glob("%s/*.png"%gt_f)
print("Total file:",len(pred_files))
pred_list = []
gt_list = []
for i in range(len(pred_files)):
    assert(os.path.basename(pred_files[i])==os.path.basename(gt_files[i]))
    pred = cv2.imread(pred_files[i],0)
    pred[pred !=0] = 1

    gt = cv2.resize(cv2.imread(gt_files[i],0), (256, 256))
    gt[gt!=0]=1

    if np.count_nonzero(gt) ==0 and np.count_nonzero(pred) ==0: continue
    pred_list.append(pred)
    gt_list.append(gt)

p_tp, p_fp, p_fn, precision_, recall_, f_score_poly = cal(pred_list,gt_list)
tp = sum(p_tp)
fp = sum(p_fp)
fn = sum(p_fn)
print("True Positive: ", tp)
print("False Positive: ", fp)
print("False Negative: ", fn)
print("Presicion: ", precision_)
print("Recall: ", recall_)
print("F1 Score Per Polygon: ", f_score_poly)