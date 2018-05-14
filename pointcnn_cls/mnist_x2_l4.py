#!/usr/bin/python3

import os
import sys
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils

load_fn = data_utils.load_cls_train_val
balance_fn = None
map_fn = None
keep_remainder = True
save_ply_fn = None

num_class = 10

sample_num = 160

batch_size = 256

num_epochs = 1024

step_val = 500

learning_rate_base = 0.01
decay_steps = 8000
decay_rate = 0.6
learning_rate_min = 0.00001

weight_decay = 1e-5

jitter = 0.0
jitter_val = 0.0

rotation_range = [0, math.pi / 18, 0, 'g']
rotation_range_val = [0, 0, 0, 'u']
rotation_order = 'rxyz'

scaling_range = [0.05, 0.05, 0.05, 'g']
scaling_range_val = [0, 0, 0, 'u']

sample_num_variance = 1 // 8
sample_num_clip = 1 // 4

x = 2

xconv_param_name = ('K', 'D', 'P', 'C', 'links')
xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                [(8, 1, -1, 16 * x, []),
                 (12, 3, -1, 32 * x, []),
                 (16, 3, 120, 64 * x, []),
                 (16, 5, 120, 128 * x, [])]]

with_global = True

fc_param_name = ('C', 'dropout_rate')
fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
             [(128 * x, 0.0),
              (64 * x, 0.5)]]

sampling = 'random'

optimizer = 'adam'
epsilon = 1e-2

data_dim = 4
use_extra_features = True
with_normal_feature = False
with_X_transformation = True
sorting_method = None

