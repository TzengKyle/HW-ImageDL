import cv2;
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage, QPixmap
import random
import pandas as pd
import cifar10
import string

from keras.preprocessing import image
from keras.applications import vgg19
from keras.utils import load_img
from keras.utils import img_to_array #tensorflow.

from torchvision import transforms
import torchvision.transforms as T
from PIL import Image

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torchvision import transforms
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt




from keras.models import load_model

from UI import Ui_MainWindow

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
result = 0
prob = 0

def load_image(filename):
	img = load_img(filename, target_size=(32, 32))
	img = img_to_array(img)
	img = img.reshape(1, 32, 32, 3)
	img = img.astype('float32')
	img = img / 255.0
	return img

def Show_Inference(img):
    img = load_image(img)
    nmodel = load_model('model_test.h5')
    result = classes[nmodel.predict(img).argmax()]
    prob = nmodel.predict(img)
    print(result)
    print(prob[0][nmodel.predict(img).argmax()])
    return result,prob[0][nmodel.predict(img).argmax()]

def Show_Train_Image():
    random.seed(12)

    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict

    file = r'cifar-10-batches-py\data_batch_1' #C:\forQT\cifar-10-python.tar\cifar-10-batches-py\data_batch_1
    data_batch_1 = unpickle(file)

    meta_file = r'cifar-10-batches-py\batches.meta'
    meta_data = unpickle(meta_file)

    images = data_batch_1['data']
    images = images.reshape(len(images),3,32,32).transpose(0,2,3,1)
    labels = data_batch_1['labels']
    label_names = meta_data['label_names']
    rows, columns = 3, 3
    imageIdForShow = np.random.randint(0, len(images), rows * columns)
    imagesForShow = images[imageIdForShow]
    labelsForShow = [labels[i] for i in imageIdForShow]

    fig=plt.figure(figsize=(10, 10)) 
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(imagesForShow[i-1])
        plt.xticks([])
        plt.yticks([])
        plt.title("{}"
                .format(label_names[labelsForShow[i-1]]))

    plt.savefig('Train_Image.png')


def Show_Model_Structure():
    model = vgg19.VGG19(weights = None, input_shape = (32, 32, 3), classes = 10)
    print(model.summary())

def Show_Data_Augmentation(out_img):
    img = out_img
    size = (100,150)
    transform1 = transforms.Compose([
        transforms.Resize(size),
        T.RandomRotation(degrees = (0,180))
    ])
    transform2 = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomResizedCrop((100,150))
    ])
    transform3 = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    print(img)

    img1 = transform1(img)
    img2 = transform2(img)
    img3 = transform3(img)
    img1.save("1.png","png")
    img2.save("2.png","png")
    img3.save("3.png","png")

    fig = plt.figure(figsize=(10, 10))
    c = 0
    for i in range(3):
        fig.add_subplot(1, 3, c+1)
        if i == 0:
            plt.imshow(img1)
        elif i == 1:
            plt.imshow(img2)
        else:
            plt.imshow(img3)
        plt.axis('off')
        c+=1

    plt.savefig('Data_Augmentation.png')
    temp_img = cv2.imread('Data_Augmentation.png')
    cv2.imshow("Data_Augmentation",temp_img)


def Show_Accuracy_And_Loss_Button():
    img1 = cv2.imread('Train_history_tra_val.png')
    img2 = cv2.imread('Train_history_loss.png')

    fig = plt.figure(figsize=(5, 5))
    c = 0
    for i in range(2):
        fig.add_subplot(1, 2, c+1)
        if i == 0:
            plt.imshow(img1)
        else:
            plt.imshow(img2)
        plt.axis('off')
        c+=1

    plt.savefig('Accuracy_And_Loss.png')

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.img = "default"

        self.ui.Load_Image_Button.clicked.connect(self.click_Load_Image_Button)
        self.ui.Show_Train_Image_Button.clicked.connect(self.click_Show_Train_Image_Button)
        self.ui.Show_Model_Structure_Button.clicked.connect(self.click_Show_Model_Structure_Button)        
        self.ui.Show_Data_Augmentation_Button.clicked.connect(self.click_Show_Data_Augmentation_Button)
        self.ui.Show_Accuracy_And_Loss_Button.clicked.connect(self.click_Show_Accuracy_And_Loss_Button)
        self.ui.Inference_Button.clicked.connect(self.click_Show_Inference_Button)
        
    def click_Load_Image_Button(self):
        filename, filetype = self.img = QFileDialog.getOpenFileName(self, "Open file", "./")
        self.img = filename
        print(filename)

        img = cv2.imread(self.img)
        self.image = cv2.imread(self.img)
        self.image = cv2.resize(self.image,(400,400), interpolation = cv2.INTER_AREA)

        height, width, channel = self.image.shape
        bytesPerline = 3 * width
        self.qimg = QImage(self.image, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.Show_Image_Label.setPixmap(QPixmap.fromImage(self.qimg))

    def click_Show_Train_Image_Button(self):
        Show_Train_Image()

        self.Train_Image = cv2.imread('Train_Image.png')
        self.Train_Image = cv2.resize(self.Train_Image,(400,400), interpolation = cv2.INTER_AREA)

        height, width, channel = self.Train_Image.shape
        bytesPerline = 3 * width
        self.qTrain_Image = QImage(self.Train_Image, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.Show_Image_Label.setPixmap(QPixmap.fromImage(self.qTrain_Image))

    def click_Show_Model_Structure_Button(self):
        Show_Model_Structure()

    def click_Show_Data_Augmentation_Button(self):
        temp_img = Image.open(self.img)
        Show_Data_Augmentation(temp_img)

    def click_Show_Accuracy_And_Loss_Button(self):
        Show_Accuracy_And_Loss_Button()
        self.Accuracy_And_Loss_Image = cv2.imread('Accuracy_And_Loss.png')
        self.Accuracy_And_Loss_Image = cv2.resize(self.Accuracy_And_Loss_Image,(400,400), interpolation = cv2.INTER_AREA)

        height, width, channel = self.Accuracy_And_Loss_Image.shape
        bytesPerline = 3 * width
        self.qAccuracy_And_Loss_Image = QImage(self.Accuracy_And_Loss_Image, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.Show_Image_Label.setPixmap(QPixmap.fromImage(self.qAccuracy_And_Loss_Image))

    def click_Show_Inference_Button(self):
        confidence, prediction = Show_Inference(self.img)

        self.image = cv2.imread(self.img)
        self.image = cv2.resize(self.image,(400,400), interpolation = cv2.INTER_AREA)

        height, width, channel = self.image.shape
        bytesPerline = 3 * width
        self.qimg = QImage(self.image, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.Show_Image_Label.setPixmap(QPixmap.fromImage(self.qimg))

        self.ui.Show_Confidence_Label.setText("Prediction label: " + str(confidence))
        self.ui.Show_Prediction_Label.setText("Confidence: " + str(prediction))
