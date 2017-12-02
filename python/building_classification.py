#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
# @file     building_classification.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2017-08-20
#
# @brief    Build and evaluate a heuristic buidling classification system using
#           Wi-Fi fingerprinting
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
OUTPUT_DIM = 3                  # number of buildings
VERBOSE = 1                     # 0 for turning off logging
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
        "-R",
        "--random_seed",
        help="random seed",
        default=0,
        type=int)
    args = parser.parse_args()

    # set variables using command-line arguments
    random_seed = args.random_seed

    ### initialize random seed generator
    np.random.seed(random_seed)
    
    train_df = pd.read_csv(path_train,header = 0) # pass header=0 to be able to replace existing names 
    train_df = train_df[:19930]
    train_AP_strengths = train_df.iloc[:,:520] #select first 520 columns

    # scale transforms data to center to the mean and component wise scale to unit variance
    train_AP_features = scale(np.asarray(train_AP_strengths).astype(float), axis=1) # convert integer to float and scale jointly (axis=1)

    # convert building IDs into one-hot encoded labels
    train_labels = np.asarray(train_df['BUILDINGID'].map(str))
    train_labels = np.asarray(pd.get_dummies(train_labels)) #labels is an array of shape 19937 x 3

    # # convert train_val_split to an array of booleans: if elem < TRAINING_RATIO = true, else: false
    # train_val_split = np.random.rand(len(train_AP_features))
    # train_val_split = train_val_split < TRAINING_RATIO

    # # We aren't given a formal testing set, so we will treat the given validation
    # # set as the testing set: We will then split our given training set into
    # # training + validation
    # train_X = train_AP_features[train_val_split]
    # train_Y = train_labels[train_val_split]
    # val_X = train_AP_features[~train_val_split]
    # val_Y = train_labels[~train_val_split]

    # turn the given validation set into a testing set
    test_df = pd.read_csv(path_validation,header = 0)
    test_AP_features = scale(np.asarray(test_df.iloc[:,0:520]).astype(float), axis=1) # convert integer to float and scale jointly (axis=1)
    test_labels = np.asarray(test_df["BUILDINGID"].map(str))
    test_labels = np.asarray(pd.get_dummies(test_labels))

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
    model.add(Dropout(dropout))
    for units in classifier_hidden_layers:
        model.add(Dense(units, activation=CLASSIFIER_ACTIVATION, use_bias=CLASSIFIER_BIAS))
        model.add(Dropout(dropout))
    model.add(Dense(OUTPUT_DIM, activation='softmax', use_bias=CLASSIFIER_BIAS))
    model.compile(optimizer=CLASSIFIER_OPTIMIZER, loss=CLASSIFIER_LOSS, metrics=['accuracy'])

    # train the model
    startTime = timer()
    model.fit(train_X, train_Y, validation_data=(val_X, val_Y), batch_size=batch_size, epochs=epochs, verbose=VERBOSE)

    # evaluate the model
    elapsedTime = timer() - startTime
    print("Model trained in %e s." % elapsedTime)
    loss, acc = model.evaluate(test_AP_features, test_labels)

    ### print out final results
    now = datetime.datetime.now()
    path_out += "_" + now.strftime("%Y%m%d-%H%M%S") + ".md"
    f = open(path_out, 'w')
    f.write("# System parameters\n")
    f.write("  - Numpy random number seed: %d\n" % random_seed)
    f.write("  - Ratio of training data to overall data: %.2f\n" % TRAINING_RATIO)
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
    f.write("# Performance\n")
    f.write("  - Loss = %e\n" % loss)
    f.write("  - Accuracy = %e\n" % acc)
    f.close()
