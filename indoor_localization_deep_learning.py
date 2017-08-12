#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
# @file     indoor_localization_deep_learning.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2017-08-11
#
# @brief    Build and evaluate a deep-learning-based indoor localization system
#           using Wi-Fi fingerprinting
#
# @remarks  This work is based on the <a href="https://keras.io/">Keras</a>-based
#           implementation of the system described in "<a
#           href="https://arxiv.org/abs/1611.02049v2">Low-effort place
#           recognition with WiFi fingerprints using deep learning</a>".
#

import os.path
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import scale
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from timeit import default_timer as timer

# set paramter values
# train_val_split = 0.7
training_ratio = 0.9            # ratio of training data to overall data
input_dim = 520
output_dim = 13                 # number of labels
epochs = 20
batch_size = 10
verbose = 1                     # 0 for turning off logging
# drs = np.arange(11)*0.05        # range of dropout rate (0.0, 0.05,...,0.5)
drs = np.array([0.2, 0.5]) # range of dropout rate (0.0, 0.05,...,0.5)
losses = []
accuracies = []
encoder_model_saved = './backups/encoder_model_saved.hdf5'
encoder_activation = 'tanh'
classifier_activation = 'relu'

# the following data use "-110" to indicate lack of AP.
path_train = "~/UJIndoorLoc/trainingData2.csv"
path_validation = "~/UJIndoorLoc/validationData2.csv"

train_df = pd.read_csv(path_train,header = 0) # pass header=0 to be able to replace existing names 
train_df = train_df[:19930]
train_AP_strengths = train_df.iloc[:,:520] #select first 520 columns

# scale transforms data to center to the mean and component wise scale to unit variance
train_AP_features = scale(np.asarray(train_AP_strengths).astype(float), axis=1) # convert integer to float and scale jointly (axis=1)

# the following two objects are actually pandas.core.series.Series objects
building_ids_str = train_df["BUILDINGID"].map(str) #convert all the building ids to strings
building_floors_str = train_df["FLOOR"].map(str) #convert all the building floors to strings

res = building_ids_str + building_floors_str #element wise concatenation of BUILDINGID+FLOOR
train_labels = np.asarray(building_ids_str + building_floors_str)

# convert labels to categorical variables, dummy_labels has type 'pandas.core.frame.DataFrame'
dummy_labels = pd.get_dummies(train_labels)

"""one hot encode the dummy_labels.
this is done because dummy_labels is a dataframe with the labels (BUILDINGID+FLOOR) 
as the column names
"""
train_labels = np.asarray(dummy_labels) #labels is an array of shape 19937 x 13. (there are 13 types of labels)

# generate len(train_AP_features) of floats in between 0 and 1
train_val_split = np.random.rand(len(train_AP_features))
# convert train_val_split to an array of booleans: if elem < 0.7 = true, else: false
# train_val_split = train_val_split < 0.70 #should contain ~70% percent true
train_val_split = train_val_split < training_ratio #should contain ~90% percent true

# We aren't given a formal testing set, so we will treat the given validation
# set as the testing set: We will then split our given training set into
# training + validation
train_X = train_AP_features[train_val_split]
train_y = train_labels[train_val_split]
val_X = train_AP_features[~train_val_split]
val_y = train_labels[~train_val_split]

# turn the given validation set into a testing set
test_df = pd.read_csv(path_validation,header = 0)
test_AP_features = scale(np.asarray(test_df.iloc[:,0:520]).astype(float), axis=1) # convert integer to float and scale jointly (axis=1)
test_labels = np.asarray(test_df["BUILDINGID"].map(str) + test_df["FLOOR"].map(str))
test_labels = np.asarray(pd.get_dummies(test_labels))

### build SAE encoder model
print("\nPart 1: buidling SAE encoder model ...")
if os.path.isfile(encoder_model_saved):
    model = load_model(encoder_model_saved)
else:
    # create a model based on stacked autoencoder (SAE)
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation=encoder_activation, use_bias=True))
    model.add(Dense(128, activation=encoder_activation, use_bias=True))
    model.add(Dense(64, activation=encoder_activation, use_bias=True))
    model.add(Dense(128, activation=encoder_activation, use_bias=True))
    model.add(Dense(256, activation=encoder_activation, use_bias=True))
    model.add(Dense(input_dim, activation=encoder_activation, use_bias=True))
    model.compile(optimizer='adam', loss='mse')

    # train the model
    model.fit(train_X, train_X, batch_size=batch_size, epochs=epochs, verbose=verbose)

    # remove the decoder part
    num_to_remove = 3
    for i in range(num_to_remove):
        model.pop()

    # set all layers (i.e., SAE encoder) to non-trainable (weights will not be updated)
    for layer in model.layers[:]:
        layer.trainable = False
        
    # save the model for later use
    model.save(encoder_model_saved)

### build and evaluate a complete model with the trained SAE encoder and a new classifier
print("\nPart 2: buidling a complete model ...")
for dr in drs:
    # append a classifier to the model
    model.add(Dense(128, activation=classifier_activation, use_bias=True))
    model.add(Dropout(dr))
    model.add(Dense(128, activation=classifier_activation, use_bias=True))
    model.add(Dropout(dr))
    model.add(Dense(output_dim, activation='softmax', use_bias=True))
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])

    # train the model
    startTime = timer()
    model.fit(train_X, train_y, validation_data=(val_X, val_y), batch_size=batch_size, epochs=epochs, verbose=verbose)

    # evaluate the model
    elapsedTime = timer() - startTime
    print("Model trained with dropout rate of %f in %e s." % (dr, elapsedTime))
    loss, acc = model.evaluate(test_AP_features, test_labels)
    losses.append(loss)
    accuracies.append(acc)

### print out final results
print("\nBuidling and evaluating a complete model completed!")
print("- Ratio of training data to overall data: %f" % training_ratio)
print("- Encoder activation: %s" % encoder_activation)
print("- Classifier activation: %s" % classifier_activation)
for i in range(len(drs)):
    print("- Dropout rate = %f: Loss = %e, Accuracy = %e" % (drs[i], losses[i], accuracies[i]))
