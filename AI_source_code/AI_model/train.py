import datetime
import glob
import itertools
import os
import random
from datetime import datetime
from os import WCOREDUMP  #이거 도대체 뭐야?

import cv2
import numpy as np
import segmentation_models as sm

import tensorflow
from keras.applications.vgg16 import VGG16
from keras.callbacks import (Callback, CSVLogger, EarlyStopping,
                             ModelCheckpoint, ReduceLROnPlateau, TensorBoard)
from keras.layers import *
from keras.layers import Concatenate, Conv2D, Input, Lambda
from keras.models import *
from keras.models import Model, model_from_json
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.keras import losses
from tensorflow.python.ops.losses.losses_impl import huber_loss
from segmentation_models import get_preprocessing
from models import model_loader
from util import custom_data_generator as custom_data_generator
from util import metrics
# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 하이퍼 파라미터 설정
data_sources = "../../" # 데이터 경로
batch_size = 2 # 배치 사이즈 설정
learning_rate = 1e-4 # 학습률 설정
epoch_parameter = 300 # 에폭 수 설정
model_name = 'unet' # 모델 이름
checkpoint_name = "UNET" # 모델이 저장될 폴더 이름

# 초기값 설정
input_size = [256, 256, 6]
channeles = 3
one_hot_label = False
data_aug = True
num_classes = 1
best_iou = 0
Highest_F1_Score = 0

# 학습 데이터 리스트
train_images1_list = glob.glob("%strain/input1/*.png" % data_sources)
train_images2_list = glob.glob("%strain/input2/*.png" % data_sources)
train_labels_list = glob.glob("%strain/mask/*.png" % data_sources)
# 평가 데이터 리스트
val_images1_list = glob.glob("%sval/input1/*.png" % data_sources)
val_images2_list = glob.glob("%sval/input2/*.png" % data_sources)
val_labels_list = glob.glob("%sval/mask/*.png" % data_sources)


z = list(zip(train_images1_list, train_images2_list, train_labels_list))

random.shuffle(z)
train_images1_list, train_images2_list, train_labels_list = (zip(*z))


BACKBONE = 'vgg19'

preprocess_input = get_preprocessing(BACKBONE)
# 학습 및 평가 데이터 수 출력
print("\n\nTraining samples = %s" % (len(train_labels_list)))
print("Validation samples = %s\n\n" % (len(val_labels_list)))

assert len(train_labels_list) == len(
    train_images1_list), "Number of images not same"
assert len(val_labels_list) == len(
    val_images1_list), "Number of images not same"

# 학습 데이터와 평가 데이터 사이에 중복된 데이터가 없는 지 확인
print("Train/Val duplication= %s\n\n" %
      (len(set(train_images1_list) & set(val_images1_list))))

train_generator = custom_data_generator.image_generator(
    train_images1_list, train_images2_list, train_labels_list, batch_size, one_hot_label, data_aug, True)
val_generator = custom_data_generator.image_generator(
    val_images1_list, val_images2_list, val_labels_list, batch_size, one_hot_label, False)

checkpoint_dir = "./checkpoints/%s/" % (checkpoint_name)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Load CNN model
model = model_loader.get_model(model_name=model_name, input_size=input_size,
                               one_hot_label=one_hot_label, num_classes=num_classes)
model_json = model.to_json()
with open(checkpoint_dir+model_name+".json", 'w') as json_file:
    json_file.write(model_json)

avaliabe_gpus = len(K.tensorflow_backend._get_available_gpus()) # K 이거 뭐야?

if avaliabe_gpus > 1:

    model_parallel = multi_gpu_model(model, gpus=avaliabe_gpus)

    print("\nTraining using %s GPUs.." % avaliabe_gpus)

else:

    model_parallel = model

tensorboard = TensorBoard(log_dir=checkpoint_dir,
                          histogram_freq=0, write_graph=True, write_images=True)
weights_path = checkpoint_dir+model_name+'_weights_{epoch:02d}.h5'


class onEachEpochCheckPoint(Callback):

    def __init__(self, model_parallel, path, model, one_hot_label=one_hot_label):

        super().__init__()
        self.path = path
        self.model_for_saving = model
        self.one_hot_label = one_hot_label

    def on_epoch_end(self, epoch, logs=None):

        BG_IU, CH_IU, BG_P, CH_P, precision_, recall_, f_score_ = metrics.calculate_IoU_Per_Epoch_1(
            model, val_images1_list, val_images2_list, val_labels_list, checkpoint_dir, epoch, True, False)
    
        global best_iou
        global Highest_F1_Score

        if f_score_ > Highest_F1_Score:

            Highest_F1_Score = f_score_
            path_ = self.path.format(epoch=epoch)

            self.model_for_saving.save_weights(path_.replace(os.path.basename(
                path_), "Highest_F1_Score_" + str(epoch) + "_" + str(f_score_) + ".h5"), overwrite=True)

        print("Epoch: %s Score On Validation Data: " % (epoch))
        print("IOU = %02f" % (CH_IU), "Precision = %02f" % (precision_),
              "Recall = %02f" % (recall_), "F1Score = %02f" % (f_score_))

# Compiler and optimizer
model_parallel.compile(
    optimizer = Adam(lr=learning_rate),
    loss=sm.losses.bce_dice_loss,
    metrics=[sm.metrics.iou_score, sm.metrics.recall, sm.metrics.f1_score],
)

# Save checkpoints and model weight
model_checkpoint = onEachEpochCheckPoint(
    model_parallel, weights_path, model, one_hot_label=one_hot_label)
csv_logger = CSVLogger(checkpoint_dir+'/log.csv', append=True, separator=';')

# Train
model_parallel.fit_generator(train_generator,
                             steps_per_epoch=int(
                                 len(train_images1_list)/batch_size),
                             epochs=epoch_parameter,
                             validation_data=val_generator,
                             validation_steps=int(
                                 len(val_labels_list)/batch_size),
                             callbacks=[model_checkpoint, tensorboard, csv_logger], use_multiprocessing=True)
