#!/usr/bin/python3
import math

num_class = 13

sample_num = 2048

batch_size = 16

num_epochs = 1024

label_weights = [1.0] * num_class

learning_rate_base = 0.001
decay_steps = 5000
decay_rate = 0.8
learning_rate_min = 1e-6
step_val = 500

weight_decay = 1e-8

jitter = 0.0
jitter_val = 0.0

rotation_range = [0, math.pi/32., 0, 'u']
rotation_range_val = [0, 0, 0, 'u']
rotation_order = 'rxyz'

scaling_range = [0.001, 0.001, 0.001, 'g']
scaling_range_val = [0, 0, 0, 'u']

sample_num_variance = 1 // 8
sample_num_clip = 1 // 4

x = 8

xconv_param_name = ('K', 'D', 'P', 'C', 'links')
xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                [(8, 1, -1, 32 * x, []),
                 (12, 2, 768, 64 * x, []),
                 (16, 2, 384, 96 * x, []),
                 (16, 4, 128, 128 * x, [])]]

with_global = True

xdconv_param_name = ('K', 'D', 'pts_layer_idx', 'qrs_layer_idx')
xdconv_params = [dict(zip(xdconv_param_name, xdconv_param)) for xdconv_param in
                 [(16, 4, 3, 3),
                  (16, 2, 2, 2),
                  (12, 2, 2, 1),
                  (8, 2, 1, 0)]]

fc_param_name = ('C', 'dropout_rate')
fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
             [(32 * x, 0.0),
              (32 * x, 0.5)]]

sampling = 'fps'

optimizer = 'adam'
epsilon = 1e-5

data_dim = 6
use_extra_features = True
with_normal_feature = False
with_X_transformation = True
sorting_method = None

keep_remainder = True
