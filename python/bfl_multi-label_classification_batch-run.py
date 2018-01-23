#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import platform
import numpy as np

if platform.system() == 'Linux':
    main_script = "./bfl_multi-label_classification.py"
else:
    main_script = "python ./bfl_multi-label_classification.py"
default_options = " ".join([
    "--sae_hidden_layers", "\"256,128,256\"",
    "--classifier_hidden_layers", "\"256\""
])

for bs in [10*x for x in range(1, 6)]:
    for dr in [0.05*x for x in range(1, 7)]:
        var_options = " ".join([
            "--batch_size", str(bs),
            "--dropout", str(dr)
        ])
        cmd = " ".join([main_script, default_options, var_options])
        print("Executing: " + cmd)
        os.system(cmd)
        print("Completed.")
