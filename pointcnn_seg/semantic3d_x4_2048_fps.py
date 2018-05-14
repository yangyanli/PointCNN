#!/usr/bin/python3

import math

num_class = 8

sample_num = 2048

batch_size = 16

num_epochs = 1024

step_val = 2000

label_weights = [1.0] * num_class

learning_rate_base = 1e-2
decay_steps = 8000
decay_rate = 0.5
learning_rate_min = 1e-5

weight_decay = 0.0

jitter = 0.0
jitter_val = 0.0

rotation_range = [math.pi / 72, math.pi, math.pi / 72, 'u']
rotation_range_val = [0, 0, 0, 'u']
rotation_order = 'rxyz'

scaling_range = [0.05, 0.05, 0.05, 'g']
scaling_range_val = [0, 0, 0, 'u']

sample_num_variance = 1 // 8
sample_num_clip = 1 // 4

x = 4

xconv_param_name = ('K', 'D', 'P', 'C', 'links')
xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                [(8, 1, -1, 32 * x, []),
                 (12, 5, 768, 64 * x, []),
                 (16, 5, 256, 96 * x, [])]]

with_global = True

xdconv_param_name = ('K', 'D', 'pts_layer_idx', 'qrs_layer_idx')
xdconv_params = [dict(zip(xdconv_param_name, xdconv_param)) for xdconv_param in
                 [(12, 2, 2, 1),
                  (8, 1, 1, 0),
                  (8, 1, 0, 0)]]

fc_param_name = ('C', 'dropout_rate')
fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
             [(32 * x, 0.0),
              (32 * x, 0.8)]]

sampling = 'fps'

optimizer = 'adam'
epsilon = 1e-2

data_dim = 7
use_extra_features = True
with_normal_feature = False
with_X_transformation = True
sorting_method = None

keep_remainder = True
