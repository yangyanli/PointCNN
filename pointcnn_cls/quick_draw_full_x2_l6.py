#!/usr/bin/python3

import os
import sys

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
import data_utils

sys.path.append(os.path.join(root_folder, 'data_conversions'))
import quick_draw_utils


def load_fn(folder_npz, _):
    return quick_draw_utils.load_fn(folder_npz, 1.0)


balance_fn = None


def map_fn(stoke, label, point_num=512):
    return quick_draw_utils.map_fn(stoke, label, point_num)


keep_remainder = True


def save_ply_fn(data_sample, folder):
    data_utils.save_ply_point_with_normal(data_sample, folder)


num_parallel_calls = 16

num_class = 345

sample_num = 512

batch_size = 256

num_epochs = 32

step_val = 20000

learning_rate_base = 1e-2
decay_steps = 200000
decay_rate = 0.7
learning_rate_min = 1e-5

weight_decay = 0.0

jitter = 0.0
jitter_val = 0.0

rotation_range = [0, 0, 0, 'u']
rotation_range_val = [0, 0, 0, 'u']
order = 'rxyz'

scaling_range = [0, [0.01], 0, 'u']
scaling_range_val = [0, [0.01], 0, 'u']

sample_num_variance = 1 // 8
sample_num_clip = 1 // 4

x = 2

xconv_param_name = ('K', 'D', 'P', 'C', 'links')
xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                [(8, 2, -1, 16 * x, []),
                 (12, 2, 192, 64 * x, []),
                 (16, 1, 64, 128 * x, []),
                 (16, 2, 64, 128 * x, []),
                 (16, 3, 64, 128 * x, []),
                 (16, 4, 64, num_class * x, [])]]

fc_param_name = ('C', 'dropout_rate')
fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
             [(num_class * x, 0.0),
              (num_class * x, 0.5)]]

sampling = 'random'
with_global = True

optimizer = 'adam'
epsilon = 1e-6
sorting_method = None

data_dim = 6
use_extra_features = False
with_X_transformation = True
