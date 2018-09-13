import pandas as pd
import numpy as np
import os
import sys
from solver import *

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.layers.core import *
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

class Network_Custom_2(Solver):
    def __init__(self, args):
        super(Network_Custom_2, self).__init__(args)
    
    def build(self, inputShape, classes, finalAct="sigmoid"):
        model = Sequential()
        
        model.add(BatchNormalization(input_shape=inputShape))
            
        model.add(Conv2D(32, kernel_size=(3, 3),padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
            
        model.add(Conv2D(128, kernel_size=(3, 3),padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
            
        model.add(Conv2D(256, kernel_size=(3, 3),padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
            
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation=finalAct))

        return model































