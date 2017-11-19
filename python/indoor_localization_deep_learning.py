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

### import modules (except keras and its backend)
import argparse
import datetime
import os
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import scale
from timeit import default_timer as timer

### set paramter values
#------------------------------------------------------------------------
# general
#------------------------------------------------------------------------
training_ratio = 0.9            # ratio of training data to overall data
input_dim = 520
output_dim = 13                 # number of labels
verbose = 1                     # 0 for turning off logging
seed = 7                        # random number seed for reproducibility
#------------------------------------------------------------------------
# stacked auto encoder (sae)
#------------------------------------------------------------------------
# sae_activation = 'tanh'
sae_activation = 'relu'
sae_bias = False
sae_optimizer = 'adam'
sae_loss = 'mse'
#------------------------------------------------------------------------
# classifier
#------------------------------------------------------------------------
# classifier_activation = 'relu'
classifier_activation = 'tanh'
classifier_bias = False
classifier_optimizer = 'adam'
# classifier_optimizer = 'rmsprop'
classifier_loss = 'categorical_crossentropy'
# dropout_rates = [0.5]           # for test
# dropout_rates = np.arange(6)*0.1  # 0.0,0.1,...,0.5
#------------------------------------------------------------------------
# input files
#------------------------------------------------------------------------
path_train = '../data/UJIIndoorLoc/trainingData2.csv' # '-110' for the lack of AP.
path_validation = '../data/UJIIndoorLoc/validationData2.csv' # ditto
#------------------------------------------------------------------------
# output files
#------------------------------------------------------------------------
path_base = '../results/' + os.path.splitext(os.path.basename(__file__))[0]
path_out =  path_base + '_out'
path_sae_model = path_base + '_sae_model.hdf5'

### initialize
np.random.seed(seed)
# losses = []
# accuracies = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-G",
        "--gpu_id",
        help="ID of GPU device to run this script; default is 0; set it to a negative number for CPU (i.e., no GPU)",
        default=0,
        type=int)
    parser.add_argument(
        "-E",
        "--epochs",
        help="number of epochs; default is 20",
        default=20,
        type=int)
    parser.add_argument(
        "-B",
        "--batch_size",
        help="batch size; default is 10",
        default=10,
        type=int)
    parser.add_argument(
        "-S",
        "--sae_hidden_layers",
        help=
        "comma-separated numbers of units in SAE hidden layers; default is '256,128,64,128,256'",
        default='256,128,64,128,256',
        type=str)
    parser.add_argument(
        "-C",
        "--classifier_hidden_layers",
        help=
        "comma-separated numbers of units in classifier hidden layers; default '' (i.e., no hidden layer)",
        default='',
        type=str)
    parser.add_argument(
        "-D",
        "--dropout",
        help=
        "dropout rate before and after classifier hidden layers; default 0.0",
        default='0.0',
        type=float)
    args = parser.parse_args()

    # set variables using command-line arguments
    gpu_id = args.gpu_id
    epochs = args.epochs
    batch_size = args.batch_size
    sae_hidden_layers = [int(i) for i in (args.sae_hidden_layers).split(',')]
    if args.classifier_hidden_layers == '':
        classifier_hidden_layers = ''
    else:
        classifier_hidden_layers = [int(i) for i in (args.classifier_hidden_layers).split(',')]
    dropout = args.dropout
        
    #------------------------------------------------------------------------
    # import keras and its backend (e.g., tensorflow)
    #------------------------------------------------------------------------
    if gpu_id >= 0:
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # supress warning messages
    import tensorflow as tf
    from keras.layers import Dense, Dropout
    from keras.models import Sequential, load_model
    
    train_df = pd.read_csv(path_train,header = 0) # pass header=0 to be able to replace existing names 
    train_df = train_df[:19930]
    train_AP_strengths = train_df.iloc[:,:520] #select first 520 columns

    # scale transforms data to center to the mean and component wise scale to unit variance
    train_AP_features = scale(np.asarray(train_AP_strengths).astype(float), axis=1) # convert integer to float and scale jointly (axis=1)

    # the following two objects are actually pandas.core.series.Series objects
    building_ids_str = train_df['BUILDINGID'].map(str) #convert all the building ids to strings
    building_floors_str = train_df['FLOOR'].map(str) #convert all the building floors to strings

    train_labels = np.asarray(building_ids_str + building_floors_str) #element wise concatenation of BUILDINGID+FLOOR

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
    x_train = train_AP_features[train_val_split]
    y_train = train_labels[train_val_split]
    x_val = train_AP_features[~train_val_split]
    y_val = train_labels[~train_val_split]

    # turn the given validation set into a testing set
    test_df = pd.read_csv(path_validation,header = 0)
    test_AP_features = scale(np.asarray(test_df.iloc[:,0:520]).astype(float), axis=1) # convert integer to float and scale jointly (axis=1)
    test_labels = np.asarray(test_df["BUILDINGID"].map(str) + test_df["FLOOR"].map(str))
    test_labels = np.asarray(pd.get_dummies(test_labels))

    ### build SAE encoder model
    print("\nPart 1: buidling SAE encoder model ...")
    if False:
    # if os.path.isfile(path_sae_model) and (os.path.getmtime(path_sae_model) > os.path.getmtime(__file__)):
        model = load_model(path_sae_model)
    else:
        # create a model based on stacked autoencoder (SAE)
        model = Sequential()
        model.add(Dense(sae_hidden_layers[0], input_dim=input_dim, activation=sae_activation, use_bias=sae_bias))
        for units in sae_hidden_layers[1:]:
            model.add(Dense(units, activation=sae_activation, use_bias=sae_bias))  
        model.add(Dense(input_dim, activation=sae_activation, use_bias=sae_bias))
        model.compile(optimizer=sae_optimizer, loss=sae_loss)

        # train the model
        model.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, verbose=verbose)

        # remove the decoder part
        num_to_remove = (len(sae_hidden_layers) + 1) // 2
        for i in range(num_to_remove):
            model.pop()

        # # set all layers (i.e., SAE encoder) to non-trainable (weights will not be updated)
        # for layer in model.layers[:]:
        #     layer.trainable = False
        
        # save the model for later use
        model.save(path_sae_model)

    ### build and evaluate a complete model with the trained SAE encoder and a new classifier
    print("\nPart 2: buidling a complete model ...")
    # for dr in dropout_rates:
    # append a classifier to the model
    model.add(Dropout(dropout))
    for units in classifier_hidden_layers:
        model.add(Dense(units, activation=classifier_activation, use_bias=classifier_bias))
        model.add(Dropout(dropout))
    model.add(Dense(output_dim, activation='softmax', use_bias=classifier_bias))
    model.compile(optimizer=classifier_optimizer, loss=classifier_loss, metrics=['accuracy'])

    # train the model
    startTime = timer()
    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs, verbose=verbose)

    # evaluate the model
    elapsedTime = timer() - startTime
    # print("Model trained with dropout rate of %f in %e s." % (dr, elapsedTime))
    print("Model trained in %e s." % elapsedTime)
    loss, accuracy = model.evaluate(test_AP_features, test_labels)
    # losses.append(loss)
    # accuracies.append(acc)

    ### print out final results
    now = datetime.datetime.now()
    path_out += "_" + now.strftime("%Y%m%d-%H%M%S") + ".org"
    f = open(path_out, 'w')
    f.write("#+STARTUP: showall\n")  # unfold everything when opening
    f.write("* System parameters\n")
    f.write("  - Ratio of training data to overall data: %.2f\n" % training_ratio)
    f.write("  - Number of epochs: %d\n" % epochs)
    f.write("  - Batch size: %d\n" % batch_size)
    f.write("  - SAE hidden layers: %d" % sae_hidden_layers[0])
    for units in sae_hidden_layers[1:]:
        f.write("-%d" % units)
    f.write("\n")
    f.write("  - SAE activation: %s\n" % sae_activation)
    f.write("  - SAE bias: %s\n" % sae_bias)
    f.write("  - SAE optimizer: %s\n" % sae_optimizer)
    f.write("  - SAE loss: %s\n" % sae_loss)
    f.write("  - Classifier hidden layers: ")
    if classifier_hidden_layers == '':
        f.write("N/A\n")
    else:
        f.write("%d" % classifier_hidden_layers[0])
        for units in classifier_hidden_layers[1:]:
            f.write("-%d" % units)
        f.write("\n")
        f.write("  - Classifier hidden layer activation: %s\n" % classifier_activation)
    f.write("  - Classifier bias: %s\n" % classifier_bias)
    f.write("  - Classifier optimizer: %s\n" % classifier_optimizer)
    f.write("  - Classifier loss: %s\n" % classifier_loss)
    f.write("  - Classifier dropout rate: %.2f\n" % dropout)
    f.write("* Performance\n")
    # for i in range(len(dropout_rates)):
    # f.write("  - Dropout rate = %.2f: Loss = %e, Accuracy = %e\n" % (dropout_rates[i], losses[i], accuracies[i]))
    f.write("  - Loss = %e\n" % loss)
    f.write("  - Accuracy = %e\n" % accuracy)
    f.close()
