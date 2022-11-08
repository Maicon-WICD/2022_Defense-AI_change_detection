import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import unet as unet

def get_model(model_name, input_size, one_hot_label, num_classes):

    if model_name == "unet":
        model = unet.unet(input_size=input_size,
                          one_hot_label=one_hot_label, num_classes=num_classes)

    return model
