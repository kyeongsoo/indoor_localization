#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
# @file     building_classifier.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2017-08-20
#           2017-12-06 (file name change)
#
# @brief    Build a DNN-based building classifier for Wi-Fi fingerprinting
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
import tensorflow as tf
from keras import optimizers
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, scale
from timeit import default_timer as timer

### global constant variables
#------------------------------------------------------------------------
# general
#------------------------------------------------------------------------
TRAINING_RATIO = 0.9  # ratio of training data to overall data
VERBOSE = 1  # 0 for turning off logging
#------------------------------------------------------------------------
# input files
#------------------------------------------------------------------------
path_train = '../data/UJIIndoorLoc/trainingData2.csv'  # '-110' for the lack of AP.
path_validation = '../data/UJIIndoorLoc/validationData2.csv'  # ditto
#------------------------------------------------------------------------
# output files
#------------------------------------------------------------------------
path_base = '../results/' + os.path.splitext(os.path.basename(__file__))[0]
path_out = path_base + '_out'
path_sae_model = path_base + '_sae_model.hdf5'


class building_classifier(Sequential):
    """DNN-based building classifier for Wi-Fi fingerprinting.

    Keyword arguments:
    input_dim -- (optional) number of APs; default is 520 (for UJIIndoorLoc dataset)
    output_dim -- (optinal) number of buildings; default is 3 (")
    sae_activation -- (optional) ; default is 'relu',
    sae_bias -- (optional) ; default is False,
    sae_optimizer -- (optional) ; default is 'adam',
    sae_loss -- (optional) ; default is 'mse',
    classifier_activation -- (optional) ; default is 'tanh',
    classifier_bias -- (optional) ; default is False,
    classifier_optimizer -- (optional) ; default is 'adam',
    classifier_loss -- (optional) ; default is 'categorical_crossentropy'
    """
    
    def __init__(self,
            input_dim=520,
             output_dim=3,
             sae_activation='relu',
             sae_bias=False,
             sae_optimizer='adam',
             sae_loss='mse',
             classifier_activation='tanh',
             classifier_bias=False,
             classifier_optimizer='adam',
             classifier_loss='categorical_crossentropy'):


        ### build stacked autoencoder (SAE) encoder model
        model = Sequential()
        model.add(
            Dense(
                sae_hidden_layers[0],
                input_dim=INPUT_DIM,
                activation=SAE_ACTIVATION,
                use_bias=SAE_BIAS))
        for units in sae_hidden_layers[1:]:
            model.add(Dense(units, activation=SAE_ACTIVATION, use_bias=SAE_BIAS))
        model.add(Dense(INPUT_DIM, activation=SAE_ACTIVATION, use_bias=SAE_BIAS))
        if SAE_OPTIMIZER == 'adam':
            opt = optimizers.Adam(lr=learning_rate)
        else:
            print("The optimizer '%s' is not supported yet." % SAE_OPTIMIZER)
            print("Now exiting ...")
            sys.exit()
        model.compile(optimizer=opt, loss=SAE_LOSS)
        model.fit(
            train_X,
            train_X,
            batch_size=batch_size,
            epochs=epochs,
            verbose=VERBOSE)  # train the SAE model
        num_to_remove = (len(sae_hidden_layers) + 1) // 2
        for i in range(num_to_remove):
            model.pop()  # remove the decoder part
        # # set all layers (i.e., SAE encoder) to non-trainable (weights will not be updated)
        # for layer in model.layers[:]:
        #     layer.trainable = False
        # # save the model for later use
        # model.save(path_sae_model)

        ### append a classifier to the SAE encoder
        model.add(Dropout(dropout))
        for units in classifier_hidden_layers:
            model.add(
                Dense(
                    units,
                    activation=CLASSIFIER_ACTIVATION,
                    use_bias=CLASSIFIER_BIAS))
        model.add(Dropout(dropout))
        model.add(
            Dense(OUTPUT_DIM, activation='softmax', use_bias=CLASSIFIER_BIAS))
        if CLASSIFIER_OPTIMIZER == 'adam':
            opt = optimizers.Adam(lr=learning_rate)
        else:
            print(
                "The optimizer '%s' is not supported yet." % CLASSIFIER_ACTIVATION)
            print("Now exiting ...")
            sys.exit()
        model.compile(optimizer=opt, loss=CLASSIFIER_LOSS, metrics=['accuracy'])

        return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-G",
        "--gpu_id",
        help=
        "ID of GPU device to run this script; default is 0; set it to a negative number for CPU (i.e., no GPU)",
        default=0,
        type=int)
    parser.add_argument(
        "-R", "--random_seed", help="random seed", default=0, type=int)
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
        default=0.0,
        type=float)
    parser.add_argument(
        "-L",
        "--learning_rate",
        help="learning rate; default 0.001",
        default=0.001,
        type=float)
    args = parser.parse_args()

    # set variables using command-line arguments
    gpu_id = args.gpu_id
    random_seed = args.random_seed
    epochs = args.epochs
    batch_size = args.batch_size
    sae_hidden_layers = [int(i) for i in (args.sae_hidden_layers).split(',')]
    if args.classifier_hidden_layers == '':
        classifier_hidden_layers = ''
    else:
        classifier_hidden_layers = [
            int(i) for i in (args.classifier_hidden_layers).split(',')
        ]
    dropout = args.dropout
    learning_rate = args.learning_rate

    ### initialize random seed generator
    np.random.seed(random_seed)
    if gpu_id >= 0:
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # supress warning messages

    train_df = pd.read_csv(
        path_train,
        header=0)  # pass header=0 to be able to replace existing names
    train_df = train_df[:19930]
    train_AP_strengths = train_df.iloc[:, :520]  #select first 520 columns

    # scale transforms data to center to the mean and component wise scale to unit variance
    train_AP_features = scale(
        np.asarray(train_AP_strengths).astype(float),
        axis=1)  # convert integer to float and scale jointly (axis=1)

    # convert building IDs into one-hot encoded labels
    train_labels = np.asarray(train_df['BUILDINGID'].map(str))
    train_labels = np.asarray(
        pd.get_dummies(train_labels))  #labels is an array of shape 19937 x 3

    # convert train_val_split to an array of booleans: if elem < TRAINING_RATIO = true, else: false
    train_val_split = np.random.rand(len(train_AP_features))
    train_val_split = train_val_split < TRAINING_RATIO

    # We aren't given a formal testing set, so we will treat the given validation
    # set as the testing set: We will then split our given training set into
    # training + validation
    train_X = train_AP_features[train_val_split]
    train_y = train_labels[train_val_split]
    val_X = train_AP_features[~train_val_split]
    val_y = train_labels[~train_val_split]

    # turn the given validation set into a testing set
    test_df = pd.read_csv(path_validation, header=0)
    test_AP_features = scale(
        np.asarray(test_df.iloc[:, 0:520]).astype(float),
        axis=1)  # convert integer to float and scale jointly (axis=1)
    test_labels = np.asarray(test_df["BUILDINGID"].map(str))
    test_labels = np.asarray(pd.get_dummies(test_labels))

    # create the model
    estimator = KerasClassifier(
        build_fn=building_classifier, epochs=200, batch_size=5, verbose=0)

    # cross validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(estimator, X, dummy_y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100,
                                         results.std() * 100))

    # # train the model
    # startTime = timer()
    # model.fit(train_X, train_y, validation_data=(val_X, val_y), batch_size=batch_size, epochs=epochs, verbose=VERBOSE)

    # # evaluate the model
    # elapsedTime = timer() - startTime
    # print("Model trained in %e s." % elapsedTime)
    # loss, acc = model.evaluate(test_AP_features, test_labels)

    # ### print out final results
    # now = datetime.datetime.now()
    # path_out += "_" + now.strftime("%Y%m%d-%H%M%S") + ".md"
    # f = open(path_out, 'w')
    # f.write("# System parameters\n")
    # f.write("  - Numpy random number seed: %d\n" % random_seed)
    # f.write("  - Ratio of training data to overall data: %.2f\n" % TRAINING_RATIO)
    # f.write("  - Number of epochs: %d\n" % epochs)
    # f.write("  - Batch size: %d\n" % batch_size)
    # f.write("  - Learning rate: %e\n" % learning_rate)
    # f.write("  - SAE hidden layers: %d" % sae_hidden_layers[0])
    # for units in sae_hidden_layers[1:]:
    #     f.write("-%d" % units)
    # f.write("\n")
    # f.write("  - SAE activation: %s\n" % SAE_ACTIVATION)
    # f.write("  - SAE bias: %s\n" % SAE_BIAS)
    # f.write("  - SAE optimizer: %s\n" % SAE_OPTIMIZER)
    # f.write("  - SAE loss: %s\n" % SAE_LOSS)
    # f.write("  - Classifier hidden layers: ")
    # if classifier_hidden_layers == '':
    #     f.write("N/A\n")
    # else:
    #     f.write("%d" % classifier_hidden_layers[0])
    #     for units in classifier_hidden_layers[1:]:
    #         f.write("-%d" % units)
    #     f.write("\n")
    #     f.write("  - Classifier hidden layer activation: %s\n" % CLASSIFIER_ACTIVATION)
    # f.write("  - Classifier bias: %s\n" % CLASSIFIER_BIAS)
    # f.write("  - Classifier optimizer: %s\n" % CLASSIFIER_OPTIMIZER)
    # f.write("  - Classifier loss: %s\n" % CLASSIFIER_LOSS)
    # f.write("  - Classifier dropout rate: %.2f\n" % dropout)
    # f.write("# Performance\n")
    # f.write("  - Loss = %e\n" % loss)
    # f.write("  - Accuracy = %e\n" % acc)
    # f.close()
