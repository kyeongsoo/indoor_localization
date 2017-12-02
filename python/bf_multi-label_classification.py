#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
# @file     bf_multi-label_classification.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2017-08-18
#
# @brief    Build and evaluate a deep-learning-based buidling-floor
#           multi-label classification system using Wi-Fi fingerprinting
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


### global constant variables
#------------------------------------------------------------------------
# general
#------------------------------------------------------------------------
INPUT_DIM = 520
OUTPUT_DIM = 8                  # number of labels
VERBOSE = 0                     # 0 for turning off logging
#------------------------------------------------------------------------
# stacked auto encoder (sae)
#------------------------------------------------------------------------
# SAE_ACTIVATION = 'tanh'
SAE_ACTIVATION = 'relu'
SAE_BIAS = False
SAE_OPTIMIZER = 'adam'
SAE_LOSS = 'mse'
#------------------------------------------------------------------------
# classifier
#------------------------------------------------------------------------
CLASSIFIER_ACTIVATION = 'relu'
#CLASSIFIER_ACTIVATION = 'tanh'
CLASSIFIER_BIAS = False
CLASSIFIER_OPTIMIZER = 'adam'
# CLASSIFIER_OPTIMIZER = 'rmsprop'
CLASSIFIER_LOSS = 'binary_crossentropy'
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-G",
        "--gpu_id",
        help="ID of GPU device to run this script; default is 0; set it to a negative number for CPU (i.e., no GPU)",
        default=0,
        type=int)
    parser.add_argument(
        "-R",
        "--random_seed",
        help="random seed",
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
        "-T",
        "--training_ratio",
        help="ratio of training data to overall data: default is 0.90",
        default=0.9,
        type=float)
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
        default=0.0,
        type=float)
    parser.add_argument(
        "--building_weight",
        help=
        "weight for building classes in classifier; default 10.0",
        default=10.0,
        type=float)
    parser.add_argument(
        "--floor_weight",
        help=
        "weight for floor classes in classifier; default 1.0",
        default=1.0,
        type=float)
    args = parser.parse_args()

    # set variables using command-line arguments
    gpu_id = args.gpu_id
    random_seed = args.random_seed
    epochs = args.epochs
    batch_size = args.batch_size
    training_ratio = args.training_ratio
    sae_hidden_layers = [int(i) for i in (args.sae_hidden_layers).split(',')]
    if args.classifier_hidden_layers == '':
        classifier_hidden_layers = ''
    else:
        classifier_hidden_layers = [int(i) for i in (args.classifier_hidden_layers).split(',')]
    dropout = args.dropout
    building_weight = args.building_weight
    floor_weight = args.floor_weight

    ### initialize random seed generator
    np.random.seed(random_seed)
    
    #------------------------------------------------------------------------
    # import keras and its backend (e.g., tensorflow)
    #------------------------------------------------------------------------
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # supress warning messages
    import tensorflow as tf
    from keras.layers import Dense, Dropout
    from keras.models import Sequential, load_model
    
    train_df = pd.read_csv(path_train, header=0) # pass header=0 to be able to replace existing names 
    train_df = train_df[:19930]
    train_AP_strengths = train_df.iloc[:,:520] #select first 520 columns

    # scale transforms data to center to the mean and component wise scale to unit variance
    train_AP_features = scale(np.asarray(train_AP_strengths).astype(float), axis=1) # convert integer to float and scale jointly (axis=1)

    # build labels
    blds = np.asarray(pd.get_dummies(train_df['BUILDINGID']))
    flrs = np.asarray(pd.get_dummies(train_df['FLOOR']))
    train_labels = np.concatenate((blds, flrs), axis=1) # labels is an array of 19937 x 8 (3 for BUILDINGID and 5 for FLOOR in one hot encoding)

    # generate len(train_AP_features) of floats in between 0 and 1
    train_val_split = np.random.rand(len(train_AP_features))
    # convert train_val_split to an array of booleans: if elem < training_ratio = true, else: false
    train_val_split = train_val_split < training_ratio

    # We aren't given a formal testing set, so we will treat the given validation
    # set as the testing set: We will then split our given training set into
    # training + validation
    train_X = train_AP_features[train_val_split]
    train_y = train_labels[train_val_split]
    val_X = train_AP_features[~train_val_split]
    val_y = train_labels[~train_val_split]

    # turn the given validation set into a testing set
    test_df = pd.read_csv(path_validation, header=0)
    test_AP_features = scale(np.asarray(test_df.iloc[:,0:520]).astype(float), axis=1) # convert integer to float and scale jointly (axis=1)

    blds = np.asarray(pd.get_dummies(test_df['BUILDINGID']))
    flrs = np.asarray(pd.get_dummies(test_df['FLOOR']))
    test_labels = np.concatenate((blds, flrs), axis=1)

    ### build SAE encoder model
    print("\nPart 1: buidling SAE encoder model ...")
    if False:
    # if os.path.isfile(path_sae_model) and (os.path.getmtime(path_sae_model) > os.path.getmtime(__file__)):
        model = load_model(path_sae_model)
    else:
        # create a model based on stacked autoencoder (SAE)
        model = Sequential()
        model.add(Dense(sae_hidden_layers[0], input_dim=INPUT_DIM, activation=SAE_ACTIVATION, use_bias=SAE_BIAS))
        for units in sae_hidden_layers[1:]:
            model.add(Dense(units, activation=SAE_ACTIVATION, use_bias=SAE_BIAS))  
        model.add(Dense(INPUT_DIM, activation=SAE_ACTIVATION, use_bias=SAE_BIAS))
        model.compile(optimizer=SAE_OPTIMIZER, loss=SAE_LOSS)

        # train the model
        model.fit(train_X, train_X, batch_size=batch_size, epochs=epochs, verbose=VERBOSE)

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
    # append a classifier to the model
    class_weight = {
        0: building_weight, 1: building_weight, 2: building_weight,  # buildings
        3: floor_weight, 4: floor_weight, 5: floor_weight, 6:floor_weight, 7: floor_weight  # floors
    }
    model.add(Dropout(dropout))
    for units in classifier_hidden_layers:
        model.add(Dense(units, activation=CLASSIFIER_ACTIVATION, use_bias=CLASSIFIER_BIAS))
        model.add(Dropout(dropout))
    model.add(Dense(OUTPUT_DIM, activation='sigmoid', use_bias=CLASSIFIER_BIAS)) # 'sigmoid' for multi-label classification
    model.compile(optimizer=CLASSIFIER_OPTIMIZER, loss=CLASSIFIER_LOSS, metrics=['accuracy'])

    # train the model
    startTime = timer()
    model.fit(train_X, train_y, validation_data=(val_X, val_y), batch_size=batch_size, epochs=epochs, class_weight=class_weight, verbose=VERBOSE)

    # evaluate the model
    elapsedTime = timer() - startTime
    print("Model trained in %e s." % elapsedTime)
    # loss, acc, acc_bld, acc_flr = model.evaluate(test_AP_features, test_labels)
    preds = model.predict(test_AP_features, batch_size=batch_size)
    n_preds = preds.shape[0]
    # acc = np.mean(np.equal(np.argmax(test_labels), np.argmax(preds)).astype(float))
    blds = preds[:, :3]
    blds = np.greater_equal(blds, np.tile(np.amax(blds, axis=1).reshape(n_preds, 1), (1, 3))).astype(int) # set maximum column to 1 and others to 0 (row-wise)
    blds_results = np.equal(np.argmax(test_labels[:, :3], axis=1), np.argmax(blds, axis=1))
    acc_bld = np.mean(blds_results.astype(float))
    # acc_bld = np.mean(np.equal(np.argmax(test_labels[:, :3], axis=1), np.argmax(blds, axis=1)).astype(float))
    flrs = preds[:,3:]
    flrs = np.greater_equal(flrs, np.tile(np.amax(flrs, axis=1).reshape(n_preds,1), (1,5))).astype(int) # set maximum column to 1 and others to 0 (row-wise)
    flrs_results = np.equal(np.argmax(test_labels[:, 3:], axis=1), np.argmax(flrs, axis=1))
    acc_flr = np.mean(flrs_results.astype(float))
    # acc_flr = np.mean(np.equal(np.argmax(test_labels[:, 3:], axis=1), np.argmax(flrs, axis=1)).astype(float))
    acc = np.mean(np.equal(blds_results, flrs_results).astype(float))
    
    ### print out final results
    now = datetime.datetime.now()
    path_out += "_" + now.strftime("%Y%m%d-%H%M%S") + ".org"
    f = open(path_out, 'w')
    f.write("#+STARTUP: showall\n")  # unfold everything when opening
    f.write("* System parameters\n")
    f.write("  - Numpy random number seed: %d\n" % random_seed)
    f.write("  - Ratio of training data to overall data: %.2f\n" % training_ratio)
    f.write("  - Number of epochs: %d\n" % epochs)
    f.write("  - Batch size: %d\n" % batch_size)
    f.write("  - SAE hidden layers: %d" % sae_hidden_layers[0])
    for units in sae_hidden_layers[1:]:
        f.write("-%d" % units)
    f.write("\n")
    f.write("  - SAE activation: %s\n" % SAE_ACTIVATION)
    f.write("  - SAE bias: %s\n" % SAE_BIAS)
    f.write("  - SAE optimizer: %s\n" % SAE_OPTIMIZER)
    f.write("  - SAE loss: %s\n" % SAE_LOSS)
    f.write("  - Classifier hidden layers: ")
    if classifier_hidden_layers == '':
        f.write("N/A\n")
    else:
        f.write("%d" % classifier_hidden_layers[0])
        for units in classifier_hidden_layers[1:]:
            f.write("-%d" % units)
        f.write("\n")
        f.write("  - Classifier hidden layer activation: %s\n" % CLASSIFIER_ACTIVATION)
    f.write("  - Classifier bias: %s\n" % CLASSIFIER_BIAS)
    f.write("  - Classifier optimizer: %s\n" % CLASSIFIER_OPTIMIZER)
    f.write("  - Classifier loss: %s\n" % CLASSIFIER_LOSS)
    f.write("  - Classifier dropout rate: %.2f\n" % dropout)
    f.write("  - Classifier class weight for buildings: %.2f\n" % building_weight)
    f.write("  - Classifier class weight for floors: %.2f\n" % floor_weight)
    f.write("* Performance\n")
    # f.write("  - Loss = %e\n" % loss)
    f.write("  - Accuracy (overall) = %e\n" % acc)
    f.write("  - Accuracy (building) = %e\n" % acc_bld)
    f.write("  - Accuracy (floor) = %e\n" % acc_flr)
    f.close()
