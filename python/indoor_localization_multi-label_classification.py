#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
# @file     indoor_localization_multi-label_classification.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2017-11-01
#
# @brief    Build and evaluate a scalable indoor localization system
#           based on Wi-Fi fingerprinting using a neural-network-based
#           multi-label classifier.
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
import math
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
OUTPUT_DIM = 101                # number of labels
VERBOSE = 1                     # 0 for turning off logging
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
path_train = '../data/UJIIndoorLoc/trainingData2.csv'           # '-110' for the lack of AP.
path_validation = '../data/UJIIndoorLoc/validationData2.csv'    # ditto
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
        "comma-separated numbers of units in classifier hidden layers; default is '128,128'",
        default='128,128',
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
        "weight for building classes in classifier; default 1.0",
        default=1.0,
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
    
    #--------------------------------------------------------------------
    # import keras and its backend (e.g., tensorflow)
    #--------------------------------------------------------------------
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
    train_AP_features = scale(np.asarray(train_df.iloc[:,0:520]).astype(float), axis=1) # convert integer to float and scale jointly (axis=1)

    # map space IDs to sequential IDs per building-floor before building labels
    blds = np.unique(train_df[['BUILDINGID']])
    flrs = np.unique(train_df[['FLOOR']])    
    for bld in blds:
        for flr in flrs:
            cond = (train_df['BUILDINGID']==bld) & (train_df['FLOOR']==flr)
            _, idx = np.unique(train_df.loc[cond, 'SPACEID'], return_inverse=True)  # refer to numpy.unique manual
            train_df.loc[cond, 'SPACEID'] = idx
    
    # build labels for multi-label classification
    blds = np.asarray(pd.get_dummies(train_df['BUILDINGID']))
    flrs = np.asarray(pd.get_dummies(train_df['FLOOR']))
    spcs = np.asarray(pd.get_dummies(train_df['SPACEID']))
    rpss = np.asarray(pd.get_dummies(train_df['RELATIVEPOSITION']))
    train_labels = np.concatenate((blds, flrs, spcs, rpss), axis=1)
    # labels is an array of 19937 x 101
    # - 3 for BUILDINGID
    # - 5 for FLOOR,
    # - 91 for SPACEID,
    # - 2 for RELATIVEPOSITION

    # split the training set into training and validation sets; we will use the
    # validation set at a testing set.
    train_val_split = np.random.rand(len(train_AP_features)) < training_ratio # mask index array
    train_X = train_AP_features[train_val_split]
    train_y = train_labels[train_val_split]
    val_X = train_AP_features[~train_val_split]
    val_y = train_labels[~train_val_split]

    ### build SAE encoder model
    print("\nPart 1: buidling an SAE encoder ...")
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

    ### build and train a complete model with the trained SAE encoder and a new classifier
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
    elapsedTime = timer() - startTime
    print("Model trained in %e s." % elapsedTime)
    
    # turn the given validation set into a testing set
    test_df = pd.read_csv(path_validation, header=0)
    test_AP_features = scale(np.asarray(test_df.iloc[:,0:520]).astype(float), axis=1) # convert integer to float and scale jointly (axis=1)
    test_utm_x = np.asarray(test_df['LONGITUDE'])
    test_utm_y = np.asarray(test_df['LATITUDE'])
    blds = np.asarray(pd.get_dummies(test_df['BUILDINGID']))
    flrs = np.asarray(pd.get_dummies(test_df['FLOOR']))
    # spcs = np.asarray(pd.get_dummies(test_df['SPACEID']))
    # rpss = np.asarray(pd.get_dummies(test_df['RELATIVEPOSITION']))
    # test_labels = np.concatenate((blds, flrs, spcs, rpss), axis=1)
    test_labels = np.concatenate((blds, flrs), axis=1)

    ### evaluate the model
    print("\nPart 3: evaluating the model ...")
    # calculate the accuracy of building and floor estimation
    # # loss, acc, acc_bld, acc_flr = model.evaluate(test_AP_features, test_labels)
    preds = model.predict(test_AP_features, batch_size=batch_size)
    n_preds = preds.shape[0]
    # acc = np.mean(np.equal(np.argmax(test_labels), np.argmax(preds)).astype(float))
    blds = preds[:, :3]
    # blds = np.greater_equal(blds, np.tile(np.amax(blds, axis=1).reshape(n_preds, 1), (1, 3))).astype(int) # set maximum column to 1 and others to 0 (row-wise)
    blds_results = np.equal(np.argmax(test_labels[:, :3], axis=1), np.argmax(blds, axis=1))
    acc_bld = np.mean(blds_results.astype(float))
    # acc_bld = np.mean(np.equal(np.argmax(test_labels[:, :3], axis=1), np.argmax(blds, axis=1)).astype(float))
    flrs = preds[:,3:8]
    # flrs = np.greater_equal(flrs, np.tile(np.amax(flrs, axis=1).reshape(n_preds,1), (1,5))).astype(int) # set maximum column to 1 and others to 0 (row-wise)
    flrs_results = np.equal(np.argmax(test_labels[:, 3:8], axis=1), np.argmax(flrs, axis=1))
    acc_flr = np.mean(flrs_results.astype(float))
    # acc_flr = np.mean(np.equal(np.argmax(test_labels[:, 3:8], axis=1), np.argmax(flrs, axis=1)).astype(float))
    acc = np.mean(np.equal(blds_results, flrs_results).astype(float))

    # calculate positioning error when building and floor are correctly estimated
    mask = np.logical_and(blds_results, flrs_results) # mask index array for correct location of building and floor
    test_utm_x = test_utm_x[mask]
    test_utm_y = test_utm_y[mask]
    blds = blds[mask]
    flrs = flrs[mask]
    spcs = (preds[mask])[:,8:99]
    rpss = (preds[mask])[:,99:101]

    n_success = len(blds)       # number of correct building and floor location
    blds = np.greater_equal(blds, np.tile(np.amax(blds, axis=1).reshape(n_success, 1), (1, 3))).astype(int) # set maximum column to 1 and others to 0 (row-wise)
    flrs = np.greater_equal(flrs, np.tile(np.amax(flrs, axis=1).reshape(n_success, 1), (1, 5))).astype(int)
    spcs = np.greater_equal(spcs, np.tile(np.amax(spcs, axis=1).reshape(n_success, 1), (1, 91))).astype(int)
    rpss = np.greater_equal(rpss, np.tile(np.amax(rpss, axis=1).reshape(n_success, 1), (1, 2))).astype(int)
    preds_labels = np.concatenate((blds, flrs, spcs, rpss), axis=1)
    n_loc_failure = 0
    sum_pos_err = 0.0
    for i in range(n_success):
        idx = np.where((train_labels == preds_labels[i]).all(axis=1)) # tuple of indexes
        if idx[0].size > 0:
            x = train_df.iloc[idx[0][0], 520] # LONGITUDE
            y = train_df.iloc[idx[0][0], 521] # LATITUDE
            sum_pos_err += math.sqrt((x-test_utm_x[i])**2 + (y-test_utm_y[i])**2)
        else:
            n_loc_failure += 1
    mean_pos_err = sum_pos_err / (n_success - n_loc_failure)
    loc_failure = n_loc_failure / n_success # rate of location estimation failure given that building and floor are correctly located

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
    f.write("  - Accuracy (building): %e\n" % acc_bld)
    f.write("  - Accuracy (floor): %e\n" % acc_flr)
    f.write("  - Success rate (building-floor): %e\n" % acc)
    f.write("  - Location estimation failure rate (given the correct building/floor): %e\n" % loc_failure)
    f.write("  - Positioning error (meter): %e\n" % mean_pos_err)
    f.close()
