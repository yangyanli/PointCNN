#!/usr/bin/python3

import os
import sys
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils

load_fn = data_utils.load_cls_train_val
map_fn = None
save_ply_fn = None

num_class = 10

sample_num = 160

batch_size = 256

num_epochs = 2048

step_val = 500

learning_rate_base = 0.01
decay_steps = 8000
decay_rate = 0.6
learning_rate_min = 0.00001

weight_decay = 0.0

jitter = 0.01
jitter_val = 0.01

rotation_range = [0, math.pi / 18, 0, 'g']
rotation_range_val = [0, 0, 0, 'u']
order = 'rxyz'

scaling_range = [0.05, 0.05, 0.05, 'g']
scaling_range_val = [0, 0, 0, 'u']

x = 2

# K, D, P, C
xconv_params = [(8, 1, -1, 16 * x),
                (8, 2, -1, 32 * x),
                (8, 4, -1, 48 * x),
                (12, 4, 120, 64 * x),
                (12, 6, 120, 80 * x)]

# C, dropout_rate
fc_params = [(64 * x, 0.0), (32 * x, 0.5)]

with_fps = False

optimizer = 'adam'
epsilon = 1e-3

data_dim = 3
with_X_transformation = True
sorting_method = None

keep_remainder = True
