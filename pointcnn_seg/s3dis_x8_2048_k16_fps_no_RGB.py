#!/usr/bin/python3

num_parts = 13

sample_num = 2048

batch_size = 16

num_epochs = 1024

label_weights = [1,1,1,1,1,1,1,1,1,1,1,1,1]

learning_rate_base = 0.005
decay_steps = 20000
decay_rate = 0.8
learning_rate_min = 1e-6

weight_decay = 0.0

jitter = 0.001
jitter_val = 0.0

scaling_range = [0.1, 0.1, 0.1, 'g']
scaling_range_val = [0, 0, 0, 'u']

x = 8

# K, D, P, C
xconv_params = [(8, 1, -1, 32 * x),
                (12, 2, 768, 32 * x),
                (16, 2, 384, 64 * x),
                (16, 6, 128, 128 * x)]

# K, D, pts_layer_idx, qrs_layer_idx
xdconv_params = [(16, 6, 3, 2),
                 (12, 4, 2, 1),
                 (8, 4, 1, 0)]

# C, dropout_rate
fc_params = [(32 * x, 0.5), (32 * x, 0.5)]

with_fps = True

optimizer = 'adam'
epsilon = 1e-3

data_dim = 6
use_extra_features = False
with_X_transformation = True
sorting_method = None

keep_remainder = True
