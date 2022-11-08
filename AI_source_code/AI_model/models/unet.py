from keras.applications.vgg16 import VGG16
from keras.layers import *
from keras.models import *
from tensorflow.python.keras import losses


def unet(pretrained_weights=None, input_size=(256, 256, 6), one_hot_label=False, num_classes=1):

    inputs = Input(shape=input_size)

    conv1 = Conv2D(filters=64, kernel_size=3, activation='relu',
                   padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(filters=64, kernel_size=3, activation='relu',
                   padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(filters=128, kernel_size=3, activation='relu',
                   padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(filters=128, kernel_size=3, activation='relu',
                   padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(filters=256, kernel_size=3, activation='relu',
                   padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(filters=256, kernel_size=3, activation='relu',
                   padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(filters=512, kernel_size=3, activation='relu',
                   padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(filters=512, kernel_size=3, activation='relu',
                   padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(filters=1024, kernel_size=3, activation='relu',
                   padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(filters=1024, kernel_size=3, activation='relu',
                   padding='same', kernel_initializer='he_normal')(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(filters=512, kernel_size=2, activation='relu',
                 padding='same', kernel_initializer='he_normal')(up6)

    merge6 = concatenate([conv4, up6], axis=3)

    conv6 = Conv2D(filters=512, kernel_size=3, activation='relu',
                   padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(filters=512, kernel_size=3, activation='relu',
                   padding='same', kernel_initializer='he_normal')(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(filters=256, kernel_size=2, activation='relu',
                 padding='same', kernel_initializer='he_normal')(up7)

    merge7 = concatenate([conv3, up7], axis=3)

    conv7 = Conv2D(filters=256, kernel_size=3, activation='relu',
                   padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(filters=256, kernel_size=3, activation='relu',
                   padding='same', kernel_initializer='he_normal')(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(filters=128, kernel_size=3, activation='relu',
                 padding='same', kernel_initializer='he_normal')(up8)

    merge8 = concatenate([conv2, up8], axis=3)

    conv8 = Conv2D(filters=128, kernel_size=3, activation='relu',
                   padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(filters=128, kernel_size=3, activation='relu',
                   padding='same', kernel_initializer='he_normal')(conv8)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Conv2D(filters=64, kernel_size=2, activation='relu',
                 padding='same', kernel_initializer='he_normal')(up9)

    merge9 = concatenate([conv1, up9], axis=3)
    
    conv9 = Conv2D(filters=64, kernel_size=3, activation='relu',
                   padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(filters=64, kernel_size=3, activation='relu',
                   padding='same', kernel_initializer='he_normal')(conv9)

    if one_hot_label:

        conv10 = Conv2D(num_classes, 1, 1, activation='softmax',
                        border_same='same')(conv9)

    else:

        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model
