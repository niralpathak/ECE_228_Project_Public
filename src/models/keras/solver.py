import matplotlib
matplotlib.use("Agg")
 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.layers.core import *
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import random
import pickle
import cv2
import os
from tqdm import tqdm
import csv
import json
import sys
import pandas as pd

FILENAME_SAVED_MODEL = "saved_model.h5"
FILENAME_SAVED_BINARIZER = "label_binarizer.p"
FILENAME_SAVED_PLOT_PDF = "info_plot.pdf"
FILENAME_SAVED_PLOT_PNG = "info_plot.png"
FILENAME_SAVED_TEST_LABELS = "test_labels.csv"

def fbeta(true_label, prediction):
    """
    Compute the F2 score based on the predictions and ground truth.

    :param true_label: Ground truth.
    :param prediction: Label predictions.
    :return: F2 score.
    """
    true_label = np.asarray(true_label)
    prediction = np.asarray(prediction)
    return fbeta_score(true_label, prediction, beta=2, average='samples')

def get_optimal_threshhold(true_label, prediction, iterations=100, size=17):
    """
    Search for the optimal threshold values based on the highest fbeta score.

    :param true_label: Ground truth.
    :param prediction: Label predictions.
    :param iterations: Number of searches for each threshold parameter.
    :param size: Size of the threshold vector, corresponding to the output class size.
    :return: Best threshold value for each class.
    """
    best_threshhold = [0.2]*size
    for t in range(size):
        best_fbeta = 0
        temp_threshhold = [0.2]*size
        for i in range(iterations):
            temp_value = i / float(iterations)
            temp_threshhold[t] = temp_value
            temp_fbeta = fbeta(true_label, prediction > temp_threshhold)
            if  temp_fbeta > best_fbeta:
                best_fbeta = temp_fbeta
                best_threshhold[t] = temp_value
    return best_threshhold

class Solver():
    
    def __init__(self, args):
        self.epochs = args["epochs"]
        self.batch_size = args["batch"]
        self.init_lr = args["init_lr"]

        self.checkpoint_dir = args["checkpoint_dir"] + "/"
        if not os.path.exists(self.checkpoint_dir):
            print("Warning: " + self.checkpoint_dir + " was created")            
            os.makedirs(self.checkpoint_dir)

        with open(self.checkpoint_dir + "args.json", 'w') as file:
            file.write(json.dumps(args))

        self.training_img_path = args["training_imgs"]
        self.training_labels_path = args["training_labels"]
        self.testing_img_path = args["testing_imgs"]
        self.sample_submission = args["sample_submission"]

        self.early_stopping = EarlyStopping(monitor="val_loss", patience=args["patience"])
        self._file_checkpoint = self.checkpoint_dir + FILENAME_SAVED_MODEL
        self.model_checkpoint = ModelCheckpoint(
            self._file_checkpoint,
            monitor="val_loss",
            save_best_only=True)

        self.callback_list = [self.early_stopping, self.model_checkpoint]
        self.image_dims = (128, 128, 3)

        self.label_map = None
        self.inv_label_map = None
        self.num_classes = None
        self.optimal_thresh = None
        self.training_mean = None
        
    def _preprocess_data_train(self, IMAGE_DIMS):
        print("[INFO] preprocess_data...")
        df_train = pd.read_csv(self.training_labels_path)
        
        flatten = lambda l: [item for sublist in l for item in sublist]
        labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

        self.label_map = {l: i for i, l in enumerate(labels)}
        self.inv_label_map = {i: l for l, i in self.label_map.items()}
        self.num_classes = len(self.label_map)

        x_train = []
        y_train = []

        for f, tags in tqdm(df_train.values, desc="Reading Training Files"):
            img = cv2.imread(self.training_img_path + "/{}.jpg".format(f))
            targets = np.zeros(self.num_classes)
            for t in tags.split(' '):
                targets[self.label_map[t]] = 1
            if img is None:
                continue
            x_train.append(cv2.resize(img, (IMAGE_DIMS[0], IMAGE_DIMS[1])))
            y_train.append(targets)
            
        y_train = np.array(y_train, np.uint8)
        x_train = np.array(x_train, np.float) / 255.

        print("[INFO] mean subtracting...")
        self.training_mean = x_train.mean(axis=(0,1,2),keepdims=1)
        x_train -= self.training_mean
        print("[INFO] finished mean subtracting...")        

        return x_train, y_train
    
    def _preprocess_data_test(self, IMAGE_DIMS):
        print("[INFO] preprocess_data...")
        df_test = pd.read_csv(self.sample_submission)
        filenames = []
        x_test = []

        for f, tags in tqdm(df_test.values, desc="Reading Testing Files"):
            img = cv2.imread(self.testing_img_path + "/{}.jpg".format(f))
            if img is None:
                continue
            x_test.append(cv2.resize(img, (IMAGE_DIMS[0], IMAGE_DIMS[1])))
            filenames.append(f)
        x_test = np.array(x_test, np.float) / 255.

        print("[INFO] mean subtracting...")
        x_test -= self.training_mean
        print("[INFO] finished mean subtracting...")  

        return x_test, filenames

    def build(self, inputShape, classes, finalAct="sigmoid"):
        raise NotImplementedError

    def train(self, IMAGE_DIMS = None):
        if IMAGE_DIMS is None:
            IMAGE_DIMS = self.image_dims
        X, Y = self._preprocess_data_train(IMAGE_DIMS)

        (trainX, validX, trainY, validY) = train_test_split(X, Y, test_size=0.2)

        aug = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=90,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)  # randomly flip images

        self.model = self.build(IMAGE_DIMS, self.num_classes, finalAct="sigmoid")

        self.model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

        H = self.model.fit_generator(
            aug.flow(trainX, trainY, batch_size=self.batch_size),
            validation_data=(validX, validY),
            steps_per_epoch=len(trainX) // self.batch_size,
            epochs=self.epochs, 
            verbose=1, 
            callbacks=self.callback_list)

        if os.path.isfile(self._file_checkpoint):
            self.model.load_weights(self._file_checkpoint)

        predY = self.model.predict(validX)
        self.optimal_thresh = get_optimal_threshhold(validY, predY, size=self.num_classes)
        print("Optimal Thresholds:")
        print(self.optimal_thresh)

        self._plot(H)

    def test(self, IMAGE_DIMS = None):
        if IMAGE_DIMS is None:
            IMAGE_DIMS = self.image_dims
        if self.model is None:
            raise Exception("Must train model first before testing")

        img_data, img_filenames = self._preprocess_data_test(IMAGE_DIMS)

        thresh = self.optimal_thresh

        img_data = np.asarray(img_data)
        predictions = self.model.predict(img_data, verbose=1)
        predictions = (predictions > thresh).astype(int)

        final_labels = {}

        for pred, filename in tqdm(zip(predictions, img_filenames)):
            _label = " ".join(self.inv_label_map[loc] for loc in pred.nonzero()[0])
            final_labels[filename] = _label
        
        fn = self.checkpoint_dir + FILENAME_SAVED_TEST_LABELS
        file_sample_csv = open(self.sample_submission, "r")
        file_final_csv = open(fn, "w")
        readCSV = csv.reader(file_sample_csv, delimiter=',')
        writeCSV = csv.writer(file_final_csv)
        
        writeCSV.writerow(["image_name","tags"])
        for row in tqdm(readCSV, desc="Writing to " + fn): 
            _filename = row[0]
            if _filename in final_labels:
                writeCSV.writerow([_filename, final_labels[_filename]])

    def _plot(self, H):
        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        N = len(H.history["loss"])

        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")

        plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
        
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper left")
        plt.savefig(self.checkpoint_dir + FILENAME_SAVED_PLOT_PDF)
        plt.savefig(self.checkpoint_dir + FILENAME_SAVED_PLOT_PNG)

        print("*************************")
        print(self.model.summary())
        print("*************************")
