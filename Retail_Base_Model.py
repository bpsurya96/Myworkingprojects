# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 18:42:10 2017

@author: sashw
"""

from collections import Counter

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense, Flatten,Dropout, BatchNormalization,Convolution2D,MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.applications.vgg19 import VGG19

import cv2
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score

input_size = 48
input_channels = 3

epochs = 50
batch_size = 128

n_folds = 5

training = True

ensemble_voting = False  # If True, use voting for model ensemble, otherwise use averaging

import os

os.chdir("D:\\a0409a00-8-dataset_dp")

df_train_data = pd.read_csv('train.csv')
df_test_data = pd.read_csv('test.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train_data['label'].values])))


label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}



def transformations(src, choice):
    rows, cols = src.shape[0:2]
    if choice == 0:
        # Rotate 90
        M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
        src = cv2.warpAffine(src,M,(cols,rows))
    if choice == 1:
        # Rotate 90 and flip horizontally
        M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
        src = cv2.warpAffine(src,M,(cols,rows))
        src = cv2.flip(src, flipCode=1)
    if choice == 2:
        # Rotate 180
        M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
        src = cv2.warpAffine(src,M,(cols,rows))
    if choice == 3:
        # Rotate 180 and flip horizontally
        M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
        src = cv2.warpAffine(src,M,(cols,rows))
        src = cv2.flip(src, flipCode=1)
    if choice == 4:
        # Rotate 90 counter-clockwise
        M = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
        src = cv2.warpAffine(src,M,(cols,rows))
    if choice == 5:
        # Rotate 90 counter-clockwise and flip horizontally
        M = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
        src = cv2.warpAffine(src,M,(cols,rows))
        src = cv2.flip(src, flipCode=1)
    return src


import random

indices     = set(range(0,len(df_train_data)))
train_index = set(random.sample(range(len(df_train_data)), 2205))
test_index  = indices- train_index

df_train = df_train_data.ix[train_index]
print('Training on {} samples'.format(len(df_train)))

df_valid = df_train_data.ix[test_index]
print('Validating on {} samples'.format(len(df_valid)))



def train_generator():
    while True:
        for start in range(0, len(df_train), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(df_train))
            df_train_batch = df_train[start:end]
            for f, tags in df_train_batch.values:
                img = cv2.imread('train_img/{}.png'.format(f))
                img = cv2.resize(img, (input_size, input_size))
                img = transformations(img, np.random.randint(6))
                targets = np.zeros(25)
                for t in tags.split(' '):
                    targets[label_map[t]] = 1
                x_batch.append(img)
                y_batch.append(targets)
            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.uint8)
            yield x_batch, y_batch
            

def valid_generator():
    while True:
        for start in range(0, len(df_valid), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(df_valid))
            df_valid_batch = df_valid[start:end]
            for f, tags in df_valid_batch.values:
                img = cv2.imread('train_img/{}.png'.format(f))
                img = cv2.resize(img, (input_size, input_size))
                img = transformations(img, np.random.randint(6))
                targets = np.zeros(25)
                for t in tags.split(' '):
                    targets[label_map[t]] = 1
                x_batch.append(img)
                y_batch.append(targets)
            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.uint8)
            yield x_batch, y_batch



model = Sequential()
model.add(Convolution2D(64, (3,3), activation='relu',input_shape = (input_size,input_size,input_channels)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(16, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(48, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(25, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


#EarlyStopping(monitor='val_loss',
#                           patience=4,
#                           verbose=1,
#                           min_delta=1e-4)

callbacks = [
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=2,
                               cooldown=2,
                               verbose=1),
             ModelCheckpoint(filepath='weights/best_weights.hdf5',
                             save_best_only=True,
                             save_weights_only=True)]

if training:
    model.fit_generator(generator=train_generator(),
                        steps_per_epoch=(len(df_train) // batch_size) + 1,
                        epochs=epochs,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=valid_generator(),
                        validation_steps=(len(df_valid) // batch_size) + 1)







































    def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
        def mf(x):
            p2 = np.zeros_like(p)
            for i in range(25):
                p2[:, i] = (p[:, i] > x[i]).astype(np.int)
            score = fbeta_score(y, p2, beta=2, average='samples')
            return score

        x = [0.2] * 25
        for i in range(25):
            best_i2 = 0
            best_score = 0
            for i2 in range(resolution):
                i2 /= float(resolution)
                x[i] = i2
                score = mf(x)
                if score > best_score:
                    best_i2 = i2
                    best_score = score
            x[i] = best_i2
            if verbose:
                print(i, best_i2, best_score)
        return x


    # Load best weights
    model.load_weights(filepath='weights/best_weights.fold_' + str(fold_count) + '.hdf5')

    p_valid = model.predict_generator(generator=valid_generator(),
                                      steps=(len(df_valid) // batch_size) + 1)

    y_valid = []
    for f, tags in df_valid.values:
        targets = np.zeros(25)
        for t in tags.split(' '):
            targets[label_map[t]] = 1
        y_valid.append(targets)
    y_valid = np.array(y_valid, np.uint8)

    # Find optimal f2 thresholds for local validation set
    thres = optimise_f2_thresholds(y_valid, p_valid, verbose=False)

    print('F2 = {}'.format(fbeta_score(y_valid, np.array(p_valid) > thres, beta=2, average='samples')))

    thres_sum += np.array(thres, np.float32)


    def test_generator(transformation):
        while True:
            for start in range(0, len(df_test_data), batch_size):
                x_batch = []
                end = min(start + batch_size, len(df_test_data))
                df_test_batch = df_test_data[start:end]
                for f in df_test_batch.values:
                    img = cv2.imread('test_img/{}.png'.format(''.join(f)))
                    img = cv2.resize(img, (input_size, input_size))
                    img = transformations(img, transformation)
                    x_batch.append(img)
                x_batch = np.array(x_batch, np.float32)
                yield x_batch

    # 6-fold TTA
    p_full_test = []
    for i in range(6):
        p_test = model.predict_generator(generator=test_generator(transformation=i),
                                         steps=(len(df_test_data) // batch_size) + 1)
        p_full_test.append(p_test)

    p_test = np.array(p_full_test[0])
    for i in range(1, 6):
        p_test += np.array(p_full_test[i])
    p_test /= 6

    y_full_test.append(p_test)

result = np.array(y_full_test[0])
if ensemble_voting:
    for f in range(len(y_full_test[0])):  # For each file
        for tag in range(17):  # For each tag
            preds = []
            for fold in range(n_folds):  # For each fold
                preds.append(y_full_test[fold][f][tag])
            pred = Counter(preds).most_common(1)[0][0]  # Most common tag prediction among folds
            result[f][tag] = pred
else:
    for fold in range(1, n_folds):
        result += np.array(y_full_test[fold])
    result /= n_folds
result = pd.DataFrame(result, columns=labels)

preds = []
thres = (thres_sum / n_folds).tolist()


predds = np.argmax(np.array(result),1)
predds

predds = pd.Series(predds).map(inv_label_map)
predds

#for i in tqdm(range(result.shape[0]), miniters=1000):
#    a = result.ix[[i]]
#    a = a.apply(lambda x: x > thres, axis=1)
#    a = a.transpose()
#    a = a.loc[a[i] == True]
#    ' '.join(list(a.index))
#    preds.append(' '.join(list(a.index)))

df_test_data['labels'] = predds
df_test_data.to_csv('submission1.csv', index=False)

pwd
