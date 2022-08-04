import os
import random
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

dir = '' # Path of Cat/Dog dataset
categories = ['Cats', 'Dogs']

Data = []  # Feature + label
for category in categories:
    label = categories.index(category)  # 0 for cat , 1 for dog
    path = os.path.join(dir, category)
    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        petImg = cv2.imread(imgpath, 0)
        # cv2.imshow('image', petImg)
        petImg = cv2.resize(petImg, (128, 64))
        fd, hog_img = hog(petImg, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,
                          multichannel=False)
        # resized_images = np.array(petImg)
        Data.append([fd, label])

        # break
    # break

print(len(Data))
random.shuffle(Data)
Features, Labels = [], []
for F, L in Data:
    Features.append(F)
    Labels.append(L)

xtrain, xtest, ytrain, ytest = train_test_split(Features, Labels, test_size=0.25)
SVM_MODEL = SVC(C=1, kernel='poly', gamma='auto')
SVM_MODEL.fit(xtrain, ytrain)
PREDICTED = SVM_MODEL.predict(xtest)
ACCURACY = SVM_MODEL.score(xtest, ytest)
print('Accuracy = ', ACCURACY*100)
