from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage, QPixmap

from UI import Ui_MainWindow



from keras import backend as K
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.applications import ResNet50
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import keras.utils as image
import keras
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import os
import numpy as np
import random
from keras.models import load_model
from keras.utils import load_img
from keras.utils import img_to_array #tensorflow.

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2

inference_batches = []
image_on_UI = []




class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.DataSet_Path = "default"
        self.img_path = "default"

        self.ui.LoadImage_Button.clicked.connect(self.click_LoadImage_Button)
        self.ui.ShowImages_Button.clicked.connect(self.click_ShowImages_Button)
        self.ui.ShowDistribution_Button.clicked.connect(self.click_ShowDistribution_Button)
        self.ui.ShowModelStructure_Button.clicked.connect(self.click_ShowModelStructure_Button)  
        self.ui.ShowComparison_Button.clicked.connect(self.click_ShowComparison_Button)
        self.ui.Inference_Button.clicked.connect(self.click_Inference_Button)
        

    def click_LoadImage_Button(self):
        global image_on_UI

        filename, filetype = self.img = QFileDialog.getOpenFileName(self, "Open file", "./")
        self.img_path = filename
        print(filename)

        image_on_UI = cv2.imread(self.img_path)
        self.image = cv2.resize(image_on_UI,(251,271), interpolation = cv2.INTER_AREA)

        height, width, channel = self.image.shape
        bytesPerline = 3 * width
        self.qimg = QImage(self.image, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.Image_Label.setPixmap(QPixmap.fromImage(self.qimg))

    def click_ShowImages_Button(self):
        global inference_batches

        #folderpath = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.DataSet_Path = "inference_dataset"

        inference_batches = keras.utils.image_dataset_from_directory(self.DataSet_Path,
                                                        labels='inferred',
            label_mode='int',
            class_names=None,
            color_mode='rgb',
            batch_size=32,
            image_size=(224, 224),
            shuffle=True,
            seed=None,
            validation_split=None,
            subset=None,
            interpolation='bilinear',
            follow_links=False,
            crop_to_aspect_ratio=False,)

        class_names = inference_batches.class_names
        print(class_names)

        plt.figure(figsize=(6, 3))
        No_cat = 1
        No_dog = 1
        for images, labels in inference_batches.take(1):
            for i in range(9):
                if(labels[i] == 0 and No_cat):
                    ax = plt.subplot(1, 2, 1)
                    plt.imshow(images[i].numpy().astype("uint8"))
                    plt.title(class_names[labels[i]])
                    plt.axis("off")
                    No_cat = 0

                if(labels[i] == 1 and No_dog):
                    ax = plt.subplot(1, 2, 2)
                    plt.imshow(images[i].numpy().astype("uint8"))
                    plt.title(class_names[labels[i]])
                    plt.axis("off")
                    No_dog = 0
        plt.show()

    def click_ShowDistribution_Button(self):
        def show_bar(img_path):
            img = cv2.imread(img_path)
            cv2.imshow("Bar", img)
        show_bar("class_distribution.png")

    def click_ShowModelStructure_Button(self):
        model = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
                input_shape=(224,224,3))
        x = model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        output_layer = Dense(1, activation='sigmoid')(x)

        model_final = Model(inputs=model.input, outputs=output_layer)
        print(model_final.summary())

    def click_ShowComparison_Button(self):
        def show_bar(img_path):
            img = cv2.imread(img_path)
            cv2.imshow("Bar", img)
        show_bar("accuracy_comparison.png")

    def click_Inference_Button(self):
        global image_on_UI

        image_on_UI = cv2.resize(image_on_UI,(224,224))
        image_on_UI = img_to_array(image_on_UI)
        image_on_UI = image_on_UI.reshape(1, 224, 224, 3)
        classes = [ 'Cat', 'Dog']
        nmodel = load_model('model\model_Sigmoid.h5')
        pred = nmodel.predict(image_on_UI)[0][0]
        print(nmodel.predict(image_on_UI))
        if(pred <= 0.5):
            self.ui.Predict_Label.setText("Prediction: Cat")
        else:
            self.ui.Predict_Label.setText("Prediction: Dog")

          

        















"""
        def img_show(out_img):
    img = cv2.resize(out_img,(251,271), interpolation = cv2.INTER_AREA)
    height, width, channel = img.shape
    bytesPerline = 3 * width
    self.qimg = QImage(img, width, height, bytesPerline, QImage.Format_RGB888) 
    self.ui.Image_Label.setPixmap(QPixmap.fromImage(self.qimg))
       
        self.ui.ShowComparison_Button.clicked.connect(self.click_ShowComparison_Button)
        self.ui.Inference_Button.clicked.connect(self.click_Inference_Button)
        """
