#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     scalable_indoor_localization.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2017-11-15
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
INPUT_DIM = 520                 #  number of APs
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
CLASSIFIER_BIAS = False
CLASSIFIER_OPTIMIZER = 'adam'
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
    # parser.add_argument(
    #     "--building_weight",
    #     help=
    #     "weight for building classes in classifier; default 1.0",
    #     default=1.0,
    #     type=float)
    # parser.add_argument(
    #     "--floor_weight",
    #     help=
    #     "weight for floor classes in classifier; default 1.0",
    #     default=1.0,
    #     type=float)
    parser.add_argument(
        "-N",
        "--neighbours",
        help="number of (nearest) neighbour locations to consider in positioning; default is 1",
        default=1,
        type=int)
    parser.add_argument(
        "--scaling",
        help=
        "scaling factor for threshold (i.e., threshold=scaling*maximum) for the inclusion of nighbour locations to consider in positioning; default is 0.0",
        default=0.0,
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
    # building_weight = args.building_weight
    # floor_weight = args.floor_weight
    N = args.neighbours
    scaling = args.scaling

    ### initialize random seed generator of numpy
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
    tf.set_random_seed(random_seed)  # initialize random seed generator of tensorflow
    from keras.layers import Dense, Dropout
    from keras.models import Sequential, load_model

    # read both train and test dataframes for consistent label formation through one-hot encoding
    train_df = pd.read_csv(path_train, header=0) # pass header=0 to be able to replace existing names
    test_df = pd.read_csv(path_validation, header=0)
    
    train_AP_features = scale(np.asarray(train_df.iloc[:,0:520]).astype(float), axis=1) # convert integer to float and scale jointly (axis=1)
    train_df['REFPOINT'] = train_df.apply(lambda row: str(int(row['SPACEID'])) + str(int(row['RELATIVEPOSITION'])), axis=1) # add a new column
    
    blds = np.unique(train_df[['BUILDINGID']])
    flrs = np.unique(train_df[['FLOOR']])
    x_avg = {}
    y_avg = {}
    for bld in blds:
        for flr in flrs:
            # map reference points to sequential IDs per building-floor before building labels
            cond = (train_df['BUILDINGID']==bld) & (train_df['FLOOR']==flr)
            _, idx = np.unique(train_df.loc[cond, 'REFPOINT'], return_inverse=True) # refer to numpy.unique manual
            train_df.loc[cond, 'REFPOINT'] = idx
            
            # calculate the average coordinates of each building/floor
            x_avg[str(bld) + '-' + str(flr)] = np.mean(train_df.loc[cond, 'LONGITUDE'])
            y_avg[str(bld) + '-' + str(flr)] = np.mean(train_df.loc[cond, 'LATITUDE'])
    
    # build labels for multi-label classification
    len_train = len(train_df)
    blds_all = np.asarray(pd.get_dummies(pd.concat([train_df['BUILDINGID'], test_df['BUILDINGID']]))) # for consistency in one-hot encoding for both dataframes
    flrs_all = np.asarray(pd.get_dummies(pd.concat([train_df['FLOOR'], test_df['FLOOR']]))) # ditto
    blds = blds_all[:len_train]
    flrs = flrs_all[:len_train]
    # blds = np.asarray(pd.get_dummies(train_df['BUILDINGID']))
    # flrs = np.asarray(pd.get_dummies(train_df['FLOOR']))
    rfps = np.asarray(pd.get_dummies(train_df['REFPOINT']))
    train_labels = np.concatenate((blds, flrs, rfps), axis=1)
    # labels is an array of 19937 x 118
    # - 3 for BUILDINGID
    # - 5 for FLOOR,
    # - 110 for REFPOINT
    OUTPUT_DIM = train_labels.shape[1]
    
    # split the training set into training and validation sets; we will use the
    # validation set at a testing set.
    train_val_split = np.random.rand(len(train_AP_features)) < training_ratio # mask index array
    x_train = train_AP_features[train_val_split]
    y_train = train_labels[train_val_split]
    x_val = train_AP_features[~train_val_split]
    y_val = train_labels[~train_val_split]

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
        model.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, verbose=VERBOSE)

        # remove the decoder part
        num_to_remove = (len(sae_hidden_layers) + 1) // 2
        for i in range(num_to_remove):
            model.pop()

        # # set all layers (i.e., SAE encoder) to non-trainable (weights will not be updated)
        # for layer in model.layers[:]:
        #     layer.trainable = False
        
        # # save the model for later use
        # model.save(path_sae_model)

    ### build and train a complete model with the trained SAE encoder and a new classifier
    print("\nPart 2: buidling a complete model ...")
    # append a classifier to the model
    # class_weight = {
    #     0: building_weight, 1: building_weight, 2: building_weight,  # buildings
    #     3: floor_weight, 4: floor_weight, 5: floor_weight, 6:floor_weight, 7: floor_weight  # floors
    # }
    model.add(Dropout(dropout))
    for units in classifier_hidden_layers:
        model.add(Dense(units, activation=CLASSIFIER_ACTIVATION, use_bias=CLASSIFIER_BIAS))
        model.add(Dropout(dropout))
    model.add(Dense(OUTPUT_DIM, activation='sigmoid', use_bias=CLASSIFIER_BIAS)) # 'sigmoid' for multi-label classification
    model.compile(optimizer=CLASSIFIER_OPTIMIZER, loss=CLASSIFIER_LOSS, metrics=['accuracy'])

    # train the model
    startTime = timer()
    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs, verbose=VERBOSE)
    # model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs, class_weight=class_weight, verbose=VERBOSE)
    elapsedTime = timer() - startTime
    print("Model trained in %e s." % elapsedTime)
    
    # turn the given validation set into a testing set
    # test_df = pd.read_csv(path_validation, header=0)
    test_AP_features = scale(np.asarray(test_df.iloc[:,0:520]).astype(float), axis=1) # convert integer to float and scale jointly (axis=1)
    x_test_utm = np.asarray(test_df['LONGITUDE'])
    y_test_utm = np.asarray(test_df['LATITUDE'])
    # blds = np.asarray(pd.get_dummies(test_df['BUILDINGID']))
    blds = blds_all[len_train:]
    # flrs = np.asarray(pd.get_dummies(test_df['FLOOR']))
    flrs = flrs_all[len_train:]

    ### evaluate the model
    print("\nPart 3: evaluating the model ...")

    # calculate the accuracy of building and floor estimation
    preds = model.predict(test_AP_features, batch_size=batch_size)
    n_preds = preds.shape[0]
    # blds_results = (np.equal(np.argmax(test_labels[:, :3], axis=1), np.argmax(preds[:, :3], axis=1))).astype(int)
    blds_results = (np.equal(np.argmax(blds, axis=1), np.argmax(preds[:, :3], axis=1))).astype(int)
    acc_bld = blds_results.mean()
    # flrs_results = (np.equal(np.argmax(test_labels[:, 3:8], axis=1), np.argmax(preds[:, 3:8], axis=1))).astype(int)
    flrs_results = (np.equal(np.argmax(flrs, axis=1), np.argmax(preds[:, 3:8], axis=1))).astype(int)
    acc_flr = flrs_results.mean()
    acc_bf = (blds_results*flrs_results).mean()
    # rfps_results = (np.equal(np.argmax(test_labels[:, 8:118], axis=1), np.argmax(preds[:, 8:118], axis=1))).astype(int)
    # acc_rfp = rfps_results.mean()
    # acc = (blds_results*flrs_results*rfps_results).mean()
    
    # calculate positioning error when building and floor are correctly estimated
    mask = np.logical_and(blds_results, flrs_results) # mask index array for correct location of building and floor
    x_test_utm = x_test_utm[mask]
    y_test_utm = y_test_utm[mask]
    blds = blds[mask]
    flrs = flrs[mask]
    rfps = (preds[mask])[:, 8:118]

    n_success = len(blds)       # number of correct building and floor location
    # blds = np.greater_equal(blds, np.tile(np.amax(blds, axis=1).reshape(n_success, 1), (1, 3))).astype(int) # set maximum column to 1 and others to 0 (row-wise)
    # flrs = np.greater_equal(flrs, np.tile(np.amax(flrs, axis=1).reshape(n_success, 1), (1, 5))).astype(int) # ditto

    n_loc_failure = 0
    sum_pos_err = 0.0
    sum_pos_err_weighted = 0.0
    idxs = np.argpartition(rfps, -N)[:, -N:]  # (unsorted) indexes of up to N nearest neighbors
    threshold = scaling*np.amax(rfps, axis=1)
    for i in range(n_success):
        xs = []
        ys = []
        ws = []
        for j in idxs[i]:
            rfp = np.zeros(110)
            rfp[j] = 1
            rows = np.where((train_labels == np.concatenate((blds[i], flrs[i], rfp))).all(axis=1)) # tuple of row indexes
            if rows[0].size > 0:
                if rfps[i][j] >= threshold[i]:
                    xs.append(train_df.loc[train_df.index[rows[0][0]], 'LONGITUDE'])
                    ys.append(train_df.loc[train_df.index[rows[0][0]], 'LATITUDE'])
                    ws.append(rfps[i][j])
        if len(xs) > 0:
            sum_pos_err += math.sqrt((np.mean(xs)-x_test_utm[i])**2 + (np.mean(ys)-y_test_utm[i])**2)
            sum_pos_err_weighted += math.sqrt((np.average(xs, weights=ws)-x_test_utm[i])**2 + (np.average(ys, weights=ws)-y_test_utm[i])**2)
        else:
            n_loc_failure += 1
            key = str(np.argmax(blds[i])) + '-' + str(np.argmax(flrs[i]))
            pos_err = math.sqrt((x_avg[key]-x_test_utm[i])**2 + (y_avg[key]-y_test_utm[i])**2)
            sum_pos_err += pos_err
            sum_pos_err_weighted += pos_err
    # mean_pos_err = sum_pos_err / (n_success - n_loc_failure)
    mean_pos_err = sum_pos_err / n_success
    # mean_pos_err_weighted = sum_pos_err_weighted / (n_success - n_loc_failure)
    mean_pos_err_weighted = sum_pos_err_weighted / n_success
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
    f.write("  - Number of neighbours: %d\n" % N)
    f.write("  - Scaling factor for threshold: %.2f\n" % scaling)
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
    # f.write("  - Classifier class weight for buildings: %.2f\n" % building_weight)
    # f.write("  - Classifier class weight for floors: %.2f\n" % floor_weight)
    f.write("* Performance\n")
    f.write("  - Accuracy (building): %e\n" % acc_bld)
    f.write("  - Accuracy (floor): %e\n" % acc_flr)
    f.write("  - Accuracy (building-floor): %e\n" % acc_bf)
    f.write("  - Location estimation failure rate (given the correct building/floor): %e\n" % loc_failure)
    f.write("  - Positioning error (meter): %e\n" % mean_pos_err)
    f.write("  - Positioning error (weighted; meter): %e\n" % mean_pos_err_weighted)
    f.close()
