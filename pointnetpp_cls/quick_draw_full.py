#!/usr/bin/python3

import os
import sys
import tensorflow as tf

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
import data_utils

sys.path.append(os.path.join(root_folder, 'data_conversions'))
import quick_draw_utils


num_class = 345

sample_num = 512

batch_size = 16

num_epochs = 256

step_val = 4000

learning_rate_base = 0.001
decay_steps = 200000
decay_rate = 0.7
learning_rate_min = 1e-6

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(decay_steps)
BN_DECAY_CLIP = 0.99

weight_decay = 1e-6

jitter = 0.0
jitter_val = 0.0

rotation_range = [0, 0, 0, 'u']
rotation_range_val = [0, 0, 0, 'u']
rotation_order = 'rxyz'

scaling_range = [0, [0.01], 0, 'u']
scaling_range_val = [0, [0.01], 0, 'u']

xconv_params = None
save_ply_fn = None

optimizer = 'adam'

data_dim = 6
use_extra_features = False
with_X_transformation = True

num_parallel_calls = 16

keep_remainder = False


def map_fn(stoke, label, point_num=512):
    return quick_draw_utils.map_fn(stoke, label, point_num)

def load_fn(folder_npz, _):
    return quick_draw_utils.load_fn(folder_npz, 1.0)


def save_ply_fn(data_sample, folder):
    data_utils.save_ply_point_with_normal(data_sample, folder)

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*batch_size,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay
