#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, Conv3D, MaxPooling3D, ZeroPadding3D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Input
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import PCA
from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Nadam
import keras.callbacks as kcallbacks
from keras.regularizers import l2
import time
import collections
from sklearn import metrics, preprocessing

from Utils import zeroPadding, normalization, doPCA, modelStatsRecord, averageAccuracy, ssrn_SS_IN


# In[2]:


def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


# In[3]:


def assignmentToIndex( assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index


# In[4]:


def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len,pos_row+ex_len+1), :]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch


# In[5]:


def sampling(proptionVal, groundTruth):              
#divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(proptionVal * len(indices))
        train[i] = indices[:-nb_val]
        test[i] = indices[-nb_val:]
#    whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
#        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices


# In[6]:


def res4_model_ss():
    model_res4 = ssrn_SS_IN.ResnetBuilder.build_resnet_8((1, img_rows, img_cols, img_channels), nb_classes)

    RMS = RMSprop(lr=0.0003)
    # Let's train the model using RMSprop
    model_res4.compile(loss='categorical_crossentropy', optimizer=RMS, metrics=['accuracy'])

    return model_res4


# In[7]:


mat_data = sio.loadmat('C:/Users/josep/Desktop/RS/residualnet_tensorflow_keras/residualnet_tensorflow_keras/SSRN-master/datasets/IN/Indian_pines_corrected.mat')
data_IN = mat_data['indian_pines_corrected']
mat_gt = sio.loadmat('C:/Users/josep/Desktop/RS/residualnet_tensorflow_keras/residualnet_tensorflow_keras/SSRN-master/datasets/IN/Indian_pines_gt.mat')
gt_IN = mat_gt['indian_pines_gt']


# In[8]:


#new_gt_IN = set_zeros(gt_IN, [1,4,7,9,13,15,16])
new_gt_IN = gt_IN

batch_size = 16
nb_classes = 16
nb_epoch = 200  #400
img_rows, img_cols = 7, 7         #27, 27
patience = 200

INPUT_DIMENSION_CONV = 200
INPUT_DIMENSION = 200
TOTAL_SIZE = 10249
VAL_SIZE = 1025

TRAIN_SIZE = 2055
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
VALIDATION_SPLIT = 0.8                      # 20% for trainnig and 80% for validation and testing
# TRAIN_NUM = 10
# TRAIN_SIZE = TRAIN_NUM * nb_classes
# TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
# VAL_SIZE = TRAIN_SIZE
img_channels = 200
PATCH_LENGTH = 3                #Patch_size (13*2+1)*(13*2+1)

data = data_IN.reshape(np.prod(data_IN.shape[:2]),np.prod(data_IN.shape[2:]))
gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]),)

data = preprocessing.scale(data)

# scaler = preprocessing.MaxAbsScaler()
# data = scaler.fit_transform(data)


# In[ ]:




