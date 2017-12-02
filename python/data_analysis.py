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


import itertools
import numpy as np
import pandas as pd


train_df = (pd.read_csv('../data/UJIIndoorLoc/trainingData2.csv', header=0))[:19930]

blds = np.unique(train_df[['BUILDINGID']])
flrs = np.unique(train_df[['FLOOR']])

max_aps = {}
keys = []
for bld in blds:
    for flr in flrs:
        df = (train_df[(train_df['BUILDINGID']==bld) & (train_df['FLOOR']==flr)]).iloc[:,:520]
        key = str(bld) + '-' + str(flr)  # dictionary key
        max_aps[key] = set(df.idxmax(axis=1))
        keys.append(key)

for pair in itertools.combinations(keys, 2):
    intersection = max_aps[pair[0]].intersection(max_aps[pair[1]])
    print("Intersection of %s and %s: %s" % (pair[0], pair[1], str(intersection)))
