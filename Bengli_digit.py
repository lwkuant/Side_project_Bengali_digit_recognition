# -*- coding: utf-8 -*-
"""
Bengli Digit Recognition 
https://www.kaggle.com/debdoot/bdrw
"""

import numpy as np 
import scipy as sp 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns

import os
os.chdir('D:/Dataset/Side_project_Bengli_digit')
print(os.listdir('./BDRW_train_1')[:5])

labels = pd.read_excel('labels.xls', header=None)
print(labels.shape)

### Load the images
pics = []
from skimage import io 
for ind, file in enumerate(labels[0].values):
    pics.append(io.imread('./BDRW_train_1/'+file+'.jpg'))

pics = np.array(pics)    
    
### Quick overview of some examples 
plt.figure(figsize=[10, 10])
for ind, pic in enumerate(pics[:25], start=1):
    plt.subplot(5, 5, ind)
    plt.imshow(pic, cmap='gray')
    plt.axis('off')
    plt.title('ID: '+labels[0][ind-1]+', Label: '+str(labels[1][ind-1]))
plt.tight_layout()

## gray scale
from skimage.color import rgb2gray
plt.figure(figsize=[10, 10])
for ind, pic in enumerate(pics[:25], start=1):
    plt.subplot(5, 5, ind)
    plt.imshow(rgb2gray(pic))
    plt.axis('off')
    plt.title('ID: '+labels[0][ind-1]+', Label: '+str(labels[1][ind-1]))
plt.tight_layout()


### split the dataset into training and test
seed = 100
np.random.seed(seed)
from sklearn.model_selection import train_test_split
train_ind = np.array(train_test_split(labels[1], stratify=labels[1], train_size=0.8,
                                      random_state=seed)[0].index)

pics_tr = pics[train_ind]
pics_test = pics[np.setdiff1d(np.arange(len(pics)), train_ind)]


### Transform each picture to gray scale
pics_tr_gray = np.copy(pics_tr)
for ind, pic in enumerate(pics_tr_gray):
    pics_tr_gray[ind] = rgb2gray(pic)

print(pics_tr_gray[1].shape)


### Thresholding: adaptive
ind = 812
from skimage import filters
pics_tr_gray_adp = filters.threshold_adaptive(pics_tr_gray[ind], 99, 'gaussian')
plt.imshow(pics_tr_gray[ind])
plt.imshow(pics_tr_gray_adp)

ind = 810
from skimage import filters
pics_tr_gray_adp = filters.threshold_adaptive(pics_tr_gray[ind], 99, 'gaussian')
plt.imshow(pics_tr_gray[ind])
plt.imshow(pics_tr_gray_adp)
# images have white and black colors

for ind, pic in enumerate(pics_tr_gray):
    pics_tr_gray[ind] = filters.threshold_adaptive(pic, 99, 'gaussian')

    
### Transform the image that has white character
for ind, pic in enumerate(pics_tr_gray):
    if np.mean(pic)>0.5:
        pics_tr_gray[ind] = ~pic*1
    else:
        pics_tr_gray[ind] = pic*1


### Resize the sizes of images to make them the same 
# using 30*30
width = 30
height = 30

ind = 125
from skimage.transform import resize
plt.imshow(resize(pics_tr_gray[ind], [height, width], mode='nearest'))

for ind, pic in enumerate(pics_tr_gray):
    pics_tr_gray[ind] = resize(pic, [height, width], mode='nearest')

### Thresholding again
for ind, pic in enumerate(pics_tr_gray):
    pics_tr_gray[ind] = filters.threshold_adaptive(pic, 99, 'gaussian')*1
    
### Modeling    
np.random.seed(seed)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam

## reshape the dataset
X = []
for pic in pics_tr_gray:
    X.extend(pic.ravel())

X = np.array(X).reshape([pics_tr_gray.shape[0], 30, 30])

## set aside 10% for validation???
for ind, pic in enumerate(pics_tr_gray):
    pics_tr_gray[ind] = pic.reshape([1, 30, 30])

X = X.reshape([-1, 1, 30, 30])
y = np_utils.to_categorical(labels[1].values[train_ind], nb_classes=10)

model = Sequential()

model.add(Convolution2D(
    nb_filter=32,
    nb_row=5,
    nb_col=5,
    border_mode='same',     
    input_shape=(1, 30, 30)    
))
model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    border_mode='same',    
))

model.add(Convolution2D(64, 5, 5, border_mode='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))

adam = Adam(lr=1e-4)

model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(X, y, nb_epoch=200, batch_size=32)


### Model Evaluation
def test_preprocess(df, threshold=99, height=30, width=30):
    
    # gray scale
    from skimage.color import rgb2gray
    
    df_gray = np.copy(df)
    for ind, pic in enumerate(df_gray):
        df_gray[ind] = rgb2gray(pic)
        
    # adaptive threshold
    from skimage import filters
    
    for ind, pic in enumerate(df_gray):
        df_gray[ind] = filters.threshold_adaptive(pic, threshold, 'gaussian')
    
    # white character
    for ind, pic in enumerate(df_gray):
        if np.mean(pic)>0.5:
            df_gray[ind] = ~pic*1
        else:
            df_gray[ind] = pic*1

    # resize
    from skimage.transform import resize
    
    for ind, pic in enumerate(df_gray):
        df_gray[ind] = resize(pic, [height, width], mode='nearest')
    
    # adaptive threshold
    for ind, pic in enumerate(df_gray):
        df_gray[ind] = filters.threshold_adaptive(pic, threshold, 'gaussian')*1
    
    # reshape
    X = []
    for pic in df_gray:
        X.extend(pic.ravel())

    X = np.array(X).reshape([df_gray.shape[0], 30, 30]).reshape([-1, 1, 30, 30])
    
    return X
    
X_test = test_preprocess(pics_test)   
y_test = labels[1].values[np.setdiff1d(np.arange(len(pics)), train_ind)]
y_true = np_utils.to_categorical(y_test, nb_classes=10)
print(' \nAccuracy:', model.evaluate(X_test, y_true)[1])

y_pred = np.argmax(model.predict(X_test), axis=1)

from sklearn.metrics import confusion_matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=[15, 15])
sns.heatmap(confusion_matrix(y_test, y_pred), square=True, xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test), annot=True, linewidths=.5)
plt.title('Confusion Matrix', fontsize=25, y=1.05)

from sklearn.metrics import precision_recall_fscore_support
print(precision_recall_fscore_support(y_test, y_pred))

ppf = precision_recall_fscore_support(y_test, y_pred)
for ind, met in enumerate(['Precision', 'Recall', 'F1']):
    print(met+': '+'Best: '+str(np.argmax(ppf[ind]))+', '+'Worst: '+str(np.argmin(ppf[ind])))



    
    
    
    
    
    
### Testing 

