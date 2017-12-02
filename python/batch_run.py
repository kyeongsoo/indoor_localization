#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

for i in range(10):
    for j in range(6):
        cmd = "./scalable_indoor_localization.py -S \"256,128,256\" -C \"64,128\" -D 0.2 -N " + str(i+1) + " --scaling " + '0.' + str(j)
        print("Executing: " + cmd)
        os.system(cmd)
        print("Completed.")
