#!/usr/bin/python3

import os
import sys
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils

load_fn = data_utils.load_cls_train_val
map_fn = None
save_ply_fn = None

num_class = 40

sample_num = 1024

batch_size = 200

num_epochs = 2048

step_val = 500

learning_rate_base = 0.01
decay_steps = 8000
decay_rate = 0.5
learning_rate_min = 1e-6

weight_decay = 1e-4

jitter = 0.0
jitter_val = 0.0

rotation_range = [0, math.pi, 0, 'u']
rotation_range_val = [0, 0, 0, 'u']
order = 'rxyz'

scaling_range = [0.05, 0.05, 0.05, 'g']
scaling_range_val = [0, 0, 0, 'u']

x = 2

xconv_param_name = ('K', 'D', 'P', 'C', 'links')
xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                [(8, 1, -1, 16 * x, []),
                (12, 2, 384, 32 * x, []),
                (16, 2, 128, 64 * x, []),
                (16, 3, 128, 128 * x, [])]]

fc_param_name = ('C', 'dropout_rate')
fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
             [(64 * x, 0.5)]]

sampling = 'random'

optimizer = 'adam'
epsilon = 1e-3

data_dim = 6
use_extra_features = False
with_X_transformation = True
sorting_method = None

keep_remainder = True
