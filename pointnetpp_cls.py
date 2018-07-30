import os
import sys
import tensorflow as tf

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'pointnetpp_cls', 'utils'))

import tf_util
from pointnet_util import pointnet_sa_module, pointnet_sa_module_msg

l3_input_shape = (16, 128, 3)
FC1_inputs_shape = (16, 1024)
FC2_inputs_shape = (16, 512)
FC3_inputs_shape = (16, 256)


class Net:
    def __init__(self, points, features, is_training, setting):
        bn_decay = setting.get_bn_decay(tf.train.get_global_step())
        l0_xyz = points
        l0_points = None
        num_class = setting.num_class

        # Set abstraction layers
        l1_xyz, l1_points = pointnet_sa_module_msg(l0_xyz, l0_points, 512, [0.1, 0.2, 0.4], [32, 64, 128],
                                                   [[32, 32, 64], [64, 64, 128], [64, 96, 128]], is_training, bn_decay,
                                                   scope='layer1')
        l2_xyz, l2_points = pointnet_sa_module_msg(l1_xyz, l1_points, 128, [0.2, 0.4, 0.8], [64, 64, 128],
                                                   [[64, 64, 128], [128, 128, 256], [128, 128, 256]], is_training,
                                                   bn_decay, scope='layer2')
        l3_xyz, l3_points, _ = pointnet_sa_module(l3_input_shape, l2_xyz, l2_points, npoint=None, radius=None,
                                                  nsample=None, mlp=[256, 512, 1024], mlp2=None, group_all=True,
                                                  is_training=is_training, bn_decay=bn_decay, scope='layer3')

        # Fully connected layers
        net = tf.reshape(l3_points, [l3_input_shape[0], -1])
        net = tf_util.fully_connected(FC1_inputs_shape, net, 512, bn=True, is_training=is_training, scope='fc1',
                                      bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training, scope='dp1')
        net = tf_util.fully_connected(FC2_inputs_shape, net, 256, bn=True, is_training=is_training, scope='fc2',
                                      bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training, scope='dp2')
        net = tf_util.fully_connected(FC3_inputs_shape, net, num_class, activation_fn=None, scope='fc3')

        self.logits = tf.expand_dims(net,axis = 1)
