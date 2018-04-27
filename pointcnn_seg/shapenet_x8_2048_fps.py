#!/usr/bin/python3

num_class = 50

sample_num = 2048

batch_size = 24

num_epochs = 1024

label_weights = []

for c in range(num_class):
    label_weights.append(1.0)

learning_rate_base = 0.01
decay_steps = 20000
decay_rate = 0.9
learning_rate_min = 0.00001

weight_decay = 0.0

jitter = 0.001
jitter_val = 0.0

scaling_range = [0.1, 0.1, 0.1, 'g']
scaling_range_val = [0, 0, 0, 'u']

x = 8

xconv_param_name = ('K', 'D', 'P', 'C', 'links')
xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                [(8, 1, -1, 32 * x, []),
                 (12, 2, 768, 32 * x, []),
                 (16, 2, 384, 64 * x, []),
                 (16, 6, 128, 128 * x, [])]]

with_global = True

xdconv_param_name = ('K', 'D', 'pts_layer_idx', 'qrs_layer_idx')
xdconv_params = [dict(zip(xdconv_param_name, xdconv_param)) for xdconv_param in
                 [(16, 6, 3, 2),
                  (12, 6, 2, 1),
                  (8, 6, 1, 0),
                  (8, 4, 0, 0)]]

fc_param_name = ('C', 'dropout_rate')
fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
             [(32 * x, 0.5),
              (32 * x, 0.5)]]

sampling = 'fps'

optimizer = 'adam'
epsilon = 1e-3

data_dim = 3
with_X_transformation = True
sorting_method = None

keep_remainder = True
