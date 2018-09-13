import pandas as pd
import numpy as np
import os
import sys
from solver import *

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.models import *
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.layers.core import *
from keras.layers import *
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.applications.vgg16 import VGG16


class Network_Transfer_VGG16(Solver):
    def __init__(self, args):
        super(Network_Transfer_VGG16, self).__init__(args)

    def build(self, inputShape, classes, finalAct="sigmoid"):
        input_tensor = Input(shape=inputShape)
        VGG = VGG16(weights='imagenet', include_top=False, input_shape=inputShape) 

        # for layer in VGG.layers:
        #     layer.trainable = False
        
        x = BatchNormalization()(input_tensor)
        x = VGG(x)
        x = Flatten()(x)
        output_tensor = Dense(classes, activation=finalAct)(x)

        model = Model(input_tensor, output_tensor)
        return model

































