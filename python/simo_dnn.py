#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     simo_dnn.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2018-02-02
#
# @brief    A scalable indoor localization system (up to reference points)
#           based on Wi-Fi fingerprinting using a single-input and multi-output
#           (SIMO)deep neural network (DNN) model.
#
# @remarks  The results will be published in a paper submitted to the <a
#           href="http://www.XXX.com/loi/ufio20">XXX</a> journal.


### import modules (except keras and its backend)
import argparse
import collections
import datetime
import os
import math
import matplotlib
matplotlib.use('Agg')           # to directly plot to a file without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import scale
from timeit import default_timer as timer


### global variables
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
#CLASSIFIER_ACTIVATION = 'tanh'
CLASSIFIER_BIAS = False
CLASSIFIER_OPTIMIZER = 'adam'
# CLASSIFIER_OPTIMIZER = 'rmsprop'
CLASSIFIER_LOSS = 'binary_crossentropy'
#------------------------------------------------------------------------
# input files
#------------------------------------------------------------------------
path_train = '../data/UJIIndoorLoc/trainingData2.csv'           # '-110' for the lack of AP.
# path_validation = '../data/UJIIndoorLoc/validationData2.csv'    # ditto


def simo_dnn(gpu_id, random_seed, epochs, batch_size, training_validation_test_ratio, sae_model_path, sae_hidden_layers, classifier_hidden_layers, dropout):
    """Multi-label classification of building, floor, and location based on Wi-Fi fingerprinting with a deep neural network

    Keyword arguments:
    gpu_id -- ID of GPU device to run this script; set it to a negative number for CPU (i.e., no GPU)
    random_seed -- a seed for random number generator
    epoch -- number of epochs
    batch_size -- batch size
    training_validation_test_ratio -- list of numbers for the ratio of training, validation, and test data to the overall data
    sae_model_path -- full path name for SAE model load & save
    sae_hidden_layers -- list of numbers of units in SAE hidden layers
    classifier_hidden_layers --list of numbers of units in classifier hidden layers
    dropout -- dropout rate before and after classifier hidden layers
    """

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
    from keras import backend as K
    from keras.layers import Dense, Dropout
    from keras.models import Sequential, load_model

    K.clear_session()           # to avoid clutter from old models / layers.
    
    train_df = pd.read_csv(path_train, header=0) # pass header=0 to be able to replace existing names
    train_AP_features = scale(np.asarray(train_df.iloc[:,0:520]).astype(float), axis=1) # convert integer to float and scale jointly (axis=1)
    train_df['REFPOINT'] = train_df.apply(lambda row: str(int(row['SPACEID'])) + str(int(row['RELATIVEPOSITION'])), axis=1) # add a new column
    
    # map reference points to sequential IDs per building & floor before building labels
    blds = np.unique(train_df[['BUILDINGID']])
    flrs = np.unique(train_df[['FLOOR']])    
    for bld in blds:
        for flr in flrs:
            cond = (train_df['BUILDINGID']==bld) & (train_df['FLOOR']==flr)
            _, idx = np.unique(train_df.loc[cond, 'REFPOINT'], return_inverse=True) # refer to numpy.unique manual
            train_df.loc[cond, 'REFPOINT'] = idx
    
    # build labels for multi-label classification
    blds = np.asarray(pd.get_dummies(train_df['BUILDINGID']))
    flrs = np.asarray(pd.get_dummies(train_df['FLOOR']))
    rfps = np.asarray(pd.get_dummies(train_df['REFPOINT']))
    train_labels = np.concatenate((blds, flrs, rfps), axis=1)
    # labels is an array of 19937 x 118
    # - 3 for BUILDINGID
    # - 5 for FLOOR,
    # - 110 for REFPOINT
    OUTPUT_DIM = train_labels.shape[1]
    
    # split the training set into training, validation, and test sets
    train_mask = np.random.rand(len(train_labels)) < training_validation_test_ratio[0] # mask index array
    x_train = train_AP_features[train_mask]
    y_train = train_labels[train_mask]
    x_tmp = train_AP_features[~train_mask]
    y_tmp = train_labels[~train_mask]
    val_mask = np.random.rand(len(y_tmp)) < training_validation_test_ratio[1] / sum(training_validation_test_ratio[1:]) # mask index array
    x_val = x_tmp[val_mask]
    y_val = y_tmp[val_mask]
    test_AP_features = x_tmp[~val_mask]
    test_labels = y_tmp[~val_mask]

    ### build SAE encoder model
    print("\nPart 1: buidling an SAE encoder ...")
    if os.path.isfile(sae_model_path) and (os.path.getmtime(sae_model_path) > os.path.getmtime(__file__)):
        # below are the workaround from oarriaga@GitHub: https://github.com/keras-team/keras/issues/4044
        model = load_model(sae_model_path, compile=False) 
        model.compile(optimizer=SAE_OPTIMIZER, loss=SAE_LOSS)
    else:
        # create a model based on stacked autoencoder (SAE)
        # note that we name each layer explicitly to avoid any conflicts with calling model.compile() later
        model = Sequential()
        model.add(Dense(sae_hidden_layers[0], input_dim=INPUT_DIM, activation=SAE_ACTIVATION, use_bias=SAE_BIAS, name='sae_input'))
        n_hl = 0
        for units in sae_hidden_layers[1:]:
            n_hl += 1
            model.add(Dense(units, activation=SAE_ACTIVATION, use_bias=SAE_BIAS, name='sae_hidden_' + str(n_hl)))  
        model.add(Dense(INPUT_DIM, activation=SAE_ACTIVATION, use_bias=SAE_BIAS, name='sae_output'))
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
        
        # save the SAE model for later use
        model.save(sae_model_path)

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
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs, verbose=VERBOSE)
    # model.fit(x_train, y_train, validation_data=(val_X, val_y), batch_size=batch_size, epochs=epochs, class_weight=class_weight, verbose=VERBOSE)
    elapsedTime = timer() - startTime
    print("Model trained in %e s." % elapsedTime)
    
    ### evaluate the model
    print("\nPart 3: evaluating the model ...")

    # calculate the accuracy of building and floor estimation
    preds = model.predict(test_AP_features, batch_size=batch_size)
    n_preds = preds.shape[0]
    blds_results = (np.equal(np.argmax(test_labels[:, :3], axis=1), np.argmax(preds[:, :3], axis=1))).astype(int)
    acc_bld = blds_results.mean()
    flrs_results = (np.equal(np.argmax(test_labels[:, 3:8], axis=1), np.argmax(preds[:, 3:8], axis=1))).astype(int)
    acc_flr = flrs_results.mean()
    acc_bf = (blds_results*flrs_results).mean()
    rfps_results = (np.equal(np.argmax(test_labels[:, 8:118], axis=1), np.argmax(preds[:, 8:118], axis=1))).astype(int)
    acc_rfp = rfps_results.mean()
    acc = (blds_results*flrs_results*rfps_results).mean()

    ### return classification accuracies
    Results = collections.namedtuple('Results', ['accuracy', 'history'])
    Accuracy = collections.namedtuple('Accuracy', ['building', 'floor', 'building_floor', 'location', 'overall'])
    results = Results(Accuracy(acc_bld, acc_flr, acc_bf, acc_rfp, acc), history)
    return results

    
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
        "--training_validation_test_ratio",
        help="comma-separated ratio of training, validation, and test data to the overall data: default is '7,2,1'",
        default='7,2,1',
        type=str)
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
    args = parser.parse_args()

    # set variables using command-line arguments
    gpu_id = args.gpu_id
    random_seed = args.random_seed
    epochs = args.epochs
    batch_size = args.batch_size
    training_validation_test_ratio = [float(i) for i in (args.training_validation_test_ratio).split(',')]
    training_validation_test_ratio /= sum(np.asarray(training_validation_test_ratio))
    sae_hidden_layers = [int(i) for i in (args.sae_hidden_layers).split(',')]
    if args.classifier_hidden_layers == '':
        classifier_hidden_layers = ''
    else:
        classifier_hidden_layers = [int(i) for i in (args.classifier_hidden_layers).split(',')]
    dropout = args.dropout
    # building_weight = args.building_weight
    # floor_weight = args.floor_weight

    # set full path and base for file names based on input parameter values
    path_base = '../results/' + os.path.splitext(os.path.basename(__file__))[0] + '/'
    file_name = path_base \
               + 'B' + "{0:d}".format(batch_size) \
               + '_T' + args.training_validation_test_ratio.replace(',', '-') \
               + '_S' + args.sae_hidden_layers.replace(',', '-')
    sae_model_path = file_name + '.hdf5'
    file_name += '_C' + args.classifier_hidden_layers.replace(',', '-') \
                 + '_D' + "{0:.2f}".format(dropout)
    # path_out =  path_base + '_out'
    # sae_model_path = path_base + '_sae_model.hdf5'
    # now = datetime.datetime.now()
    # path_out += "_" + now.strftime("%Y%m%d-%H%M%S") + ".org"
    
    ### call simo_dnn()
    results = simo_dnn(
        gpu_id,
        random_seed,
        epochs,
        batch_size,
        training_validation_test_ratio,
        sae_model_path,
        sae_hidden_layers,
        classifier_hidden_layers,
        dropout)
    
    ### print out final results
    f = open(file_name + '.org', 'w')
    f.write("#+STARTUP: showall\n")  # unfold everything when opening
    f.write("* System parameters\n")
    f.write("  - Numpy random number seed: %d\n" % random_seed)
    f.write("  - Ratio of training data to overall data: %.2f\n" % training_validation_test_ratio[0])
    f.write("  - Ratio of validation data to overall data: %.2f\n" % training_validation_test_ratio[1])
    f.write("  - Ratio of test data to overall data: %.2f\n" % training_validation_test_ratio[2])
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
    # f.write("  - Classifier class weight for buildings: %.2f\n" % building_weight)
    # f.write("  - Classifier class weight for floors: %.2f\n" % floor_weight)
    f.write("* Performance\n")
    # f.write("  - Loss = %e\n" % loss)
    f.write("  - Accuracy (building): %e\n" % results.accuracy.building)
    f.write("  - Accuracy (floor): %e\n" % results.accuracy.floor)
    f.write("  - Accuracy (building_floor): %e\n" % results.accuracy.building_floor)
    f.write("  - Accuracy (location): %e\n" % results.accuracy.location)
    f.write("  - Accuracy (overall): %e\n" % results.accuracy.overall)
    f.close()

    ### plot history of accuracy during the training and validation
    plt.plot(results.history.history['acc'])
    plt.plot(results.history.history['val_acc'])
    # plt.title('model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Traininig', 'Validation'], loc='upper left')
    plt.savefig(os.path.expanduser(file_name) + '.pdf', format='pdf')
    plt.close('all')
