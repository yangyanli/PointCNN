#!/usr/bin/python3
"""Training and Validation On Segmentation Task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import shutil
import argparse
import importlib
import data_utils
import numpy as np
import pointfly as pf
import tensorflow as tf
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', '-t', help='Path to training set ground truth (.txt)', required=True)
    parser.add_argument('--filelist_val', '-v', help='Path to validation set ground truth (.txt)', required=True)
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load')
    parser.add_argument('--save_folder', '-s', help='Path to folder for saving check points and summary', required=True)
    parser.add_argument('--model', '-m', help='Model to use', required=True)
    parser.add_argument('--setting', '-x', help='Setting to use', required=True)
    args = parser.parse_args()

    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = os.path.join(args.save_folder, '%s_%s_%d_%s' % (args.model, args.setting, os.getpid(), time_string))
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    sys.stdout = open(os.path.join(root_folder, 'log.txt'), 'w')

    print('PID:', os.getpid())

    print(args)

    model = importlib.import_module(args.model)
    setting_path = os.path.join(os.path.dirname(__file__), args.model)
    sys.path.append(setting_path)
    setting = importlib.import_module(args.setting)

    num_epochs = setting.num_epochs
    batch_size = setting.batch_size
    sample_num = setting.sample_num
    step_val = 500
    num_class = setting.num_class
    label_weights_list = setting.label_weights
    scaling_range = setting.scaling_range
    scaling_range_val = setting.scaling_range_val
    jitter = setting.jitter
    jitter_val = setting.jitter_val

    # Prepare inputs
    print('{}-Preparing datasets...'.format(datetime.now()))
    data_train, _, data_num_train, label_train = data_utils.load_seg(args.filelist)
    data_val, _, data_num_val, label_val = data_utils.load_seg(args.filelist_val)

    # shuffle
    data_train, data_num_train, label_train = \
        data_utils.grouped_shuffle([data_train, data_num_train, label_train])

    num_train = data_train.shape[0]
    point_num = data_train.shape[1]
    num_val = data_val.shape[0]
    print('{}-{:d}/{:d} training/validation samples.'.format(datetime.now(), num_train, num_val))
    batch_num = (num_train * num_epochs + batch_size - 1) // batch_size
    print('{}-{:d} training batches.'.format(datetime.now(), batch_num))
    batch_num_val = math.ceil(num_val / batch_size)
    print('{}-{:d} testing batches per test.'.format(datetime.now(), batch_num_val))

    ######################################################################
    # Placeholders
    indices = tf.placeholder(tf.int32, shape=(None, None, 2), name="indices")
    xforms = tf.placeholder(tf.float32, shape=(None, 3, 3), name="xforms")
    rotations = tf.placeholder(tf.float32, shape=(None, 3, 3), name="rotations")
    jitter_range = tf.placeholder(tf.float32, shape=(1), name="jitter_range")
    global_step = tf.Variable(0, trainable=False, name='global_step')
    is_training = tf.placeholder(tf.bool, name='is_training')

    pts_fts = tf.placeholder(tf.float32, shape=(None, point_num, setting.data_dim), name='pts_fts')
    labels_seg = tf.placeholder(tf.int64, shape=(None, point_num), name='labels_seg')
    labels_weights = tf.placeholder(tf.float32, shape=(None, point_num), name='labels_weights')

    ######################################################################
    features_augmented = None
    if setting.data_dim > 3:
        points, features = tf.split(pts_fts, [3, setting.data_dim - 3], axis=-1, name='split_points_features')
        if setting.use_extra_features:
            features_sampled = tf.gather_nd(features, indices=indices, name='features_sampled')
            if setting.with_normal_feature:
                if setting.data_dim < 6:
                    print('Only 3D normals are supported!')
                    exit()
                elif setting.data_dim == 6:
                    features_augmented = pf.augment(features_sampled, rotations)
                else:
                    normals, rest = tf.split(features_sampled, [3, setting.data_dim - 6])
                    normals_augmented = pf.augment(normals, rotations)
                    features_augmented = tf.concat([normals_augmented, rest], axis=-1)
            else:
                features_augmented = features_sampled
    else:
        points = pts_fts

    points_sampled = tf.gather_nd(points, indices=indices, name='points_sampled')
    points_augmented = pf.augment(points_sampled, xforms, jitter_range)
    labels_sampled = tf.gather_nd(labels_seg, indices=indices, name='labels_sampled')
    labels_weights_sampled = tf.gather_nd(labels_weights, indices=indices, name='labels_weight_sampled')

    net = model.Net(points_augmented, features_augmented, num_class, is_training, setting)
    logits = net.logits
    probs = tf.nn.softmax(logits, name='probs')
    _, predictions = tf.nn.top_k(probs, name='predictions')

    loss_op = tf.losses.sparse_softmax_cross_entropy(labels=labels_sampled, logits=logits,
                                                     weights=labels_weights_sampled)
    loss_mean_op, loss_mean_update_op = tf.metrics.mean(loss_op)
    t_1_acc_op, t_1_acc_update_op = tf.metrics.precision_at_k(labels_sampled, logits, 1)
    t_1_per_class_acc_op, t_1_per_class_acc_update_op = tf.metrics.mean_per_class_accuracy(labels_sampled,
                                                                                           predictions,
                                                                                           setting.num_class)
    _ = tf.summary.scalar('loss/train_seg', tensor=loss_mean_op, collections=['train'])
    _ = tf.summary.scalar('t_1_acc/train_seg', tensor=t_1_acc_op, collections=['train'])
    _ = tf.summary.scalar('t_1_per_class_acc/train', tensor=t_1_per_class_acc_op, collections=['train'])

    with tf.name_scope('val_metrics'):
        loss_val_op, loss_val_update_op = tf.metrics.mean(loss_op)
        t_1_acc_val_op, t_1_acc_val_update_op = tf.metrics.precision_at_k(labels_sampled, logits, 1)
        t_1_per_class_acc_val_op, t_1_per_class_acc_val_update_op = \
            tf.metrics.mean_per_class_accuracy(labels_sampled, predictions, setting.num_class)
    reset_val_metrics_op = tf.variables_initializer([var for var in tf.local_variables()
                                                     if var.name.split('/')[0] == 'val_metrics'])
    _ = tf.summary.scalar('loss/val_seg', tensor=loss_val_op, collections=['val'])
    _ = tf.summary.scalar('t_1_acc/val_seg', tensor=t_1_acc_val_op, collections=['val'])
    _ = tf.summary.scalar('t_1_per_class_acc/val_seg', tensor=t_1_per_class_acc_val_op, collections=['val'])

    lr_exp_op = tf.train.exponential_decay(setting.learning_rate_base, global_step, setting.decay_steps,
                                           setting.decay_rate, staircase=True)
    lr_clip_op = tf.maximum(lr_exp_op, setting.learning_rate_min)
    _ = tf.summary.scalar('learning_rate', tensor=lr_clip_op, collections=['train'])
    reg_loss = setting.weight_decay * tf.losses.get_regularization_loss()
    if setting.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_clip_op, epsilon=setting.epsilon)
    elif setting.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr_clip_op, momentum=setting.momentum, use_nesterov=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss_op + reg_loss, global_step=global_step)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver(max_to_keep=None)

    # backup all code
    code_folder = os.path.abspath(os.path.dirname(__file__))
    shutil.copytree(code_folder, os.path.join(root_folder, os.path.basename(code_folder)))

    folder_ckpt = os.path.join(root_folder, 'ckpts')
    if not os.path.exists(folder_ckpt):
        os.makedirs(folder_ckpt)

    folder_summary = os.path.join(root_folder, 'summary')
    if not os.path.exists(folder_summary):
        os.makedirs(folder_summary)

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

    with tf.Session() as sess:
        summaries_op = tf.summary.merge_all('train')
        summaries_val_op = tf.summary.merge_all('val')
        summary_writer = tf.summary.FileWriter(folder_summary, sess.graph)

        sess.run(init_op)

        # Load the model
        if args.load_ckpt is not None:
            saver.restore(sess, args.load_ckpt)
            print('{}-Checkpoint loaded from {}!'.format(datetime.now(), args.load_ckpt))

        for batch_idx_train in range(batch_num):
            if (batch_idx_train % step_val == 0 and (batch_idx_train != 0 or args.load_ckpt is not None)) \
                    or batch_idx_train == batch_num - 1:
                ######################################################################
                # Validation
                filename_ckpt = os.path.join(folder_ckpt, 'iter')
                saver.save(sess, filename_ckpt, global_step=global_step)
                print('{}-Checkpoint saved to {}!'.format(datetime.now(), filename_ckpt))

                sess.run(reset_val_metrics_op)
                for batch_val_idx in range(batch_num_val):
                    start_idx = batch_size * batch_val_idx
                    end_idx = min(start_idx + batch_size, num_val)
                    batch_size_val = end_idx - start_idx
                    points_batch = data_val[start_idx:end_idx, ...]
                    points_num_batch = data_num_val[start_idx:end_idx, ...]
                    labels_batch = label_val[start_idx:end_idx, ...]
                    weights_batch = np.array(label_weights_list)[label_val[start_idx:end_idx, ...]]

                    xforms_np, rotations_np = pf.get_xforms(batch_size_val, scaling_range=scaling_range_val)
                    sess.run([loss_val_update_op, t_1_acc_val_update_op, t_1_per_class_acc_val_update_op],
                             feed_dict={
                                 pts_fts: points_batch,
                                 indices: pf.get_indices(batch_size_val, sample_num, points_num_batch),
                                 xforms: xforms_np,
                                 rotations: rotations_np,
                                 jitter_range: np.array([jitter_val]),
                                 labels_seg: labels_batch,
                                 labels_weights: weights_batch,
                                 is_training: False,
                             })

                loss_val, t_1_acc_val, t_1_per_class_acc_val, summaries_val = sess.run(
                    [loss_val_op, t_1_acc_val_op, t_1_per_class_acc_val_op, summaries_val_op])
                summary_writer.add_summary(summaries_val, batch_idx_train)
                print('{}-[Val  ]-Average:      Loss: {:.4f}  T-1 Acc: {:.4f}  T-1 mAcc: {:.4f}'
                      .format(datetime.now(), loss_val, t_1_acc_val, t_1_per_class_acc_val))
                sys.stdout.flush()
                ######################################################################

            ######################################################################
            # Training
            start_idx = (batch_size * batch_idx_train) % num_train
            end_idx = min(start_idx + batch_size, num_train)
            batch_size_train = end_idx - start_idx
            points_batch = data_train[start_idx:end_idx, ...]
            points_num_batch = data_num_train[start_idx:end_idx, ...]
            labels_batch = label_train[start_idx:end_idx, ...]
            weights_batch = np.array(label_weights_list)[labels_batch]

            if start_idx + batch_size_train == num_train:
                data_train, data_num_train, label_train = \
                    data_utils.grouped_shuffle([data_train, data_num_train, label_train])

            offset = int(random.gauss(0, sample_num // 8))
            offset = max(offset, -sample_num // 4)
            offset = min(offset, sample_num // 4)
            sample_num_train = sample_num + offset
            xforms_np, rotations_np = pf.get_xforms(batch_size_train, scaling_range=scaling_range)
            sess.run([train_op, loss_mean_update_op, t_1_acc_update_op, t_1_per_class_acc_update_op],
                     feed_dict={
                         pts_fts: points_batch,
                         indices: pf.get_indices(batch_size_train, sample_num_train,
                                                 points_num_batch),
                         xforms: xforms_np,
                         rotations: rotations_np,
                         jitter_range: np.array([jitter]),
                         labels_seg: labels_batch,
                         labels_weights: weights_batch,
                         is_training: True,
                     })
            loss, t_1_acc, t_1_per_class_acc, summaries = sess.run([loss_mean_op,
                                                                    t_1_acc_op,
                                                                    t_1_per_class_acc_op,
                                                                    summaries_op])
            summary_writer.add_summary(summaries, batch_idx_train)
            print('{}-[Train]-Iter: {:06d}  Loss: {:.4f}  T-1 Acc: {:.4f}  T-1 mAcc: {:.4f}'
                  .format(datetime.now(), batch_idx_train, loss, t_1_acc, t_1_per_class_acc))
            sys.stdout.flush()
            ######################################################################
        print('{}-Done!'.format(datetime.now()))


if __name__ == '__main__':
    main()
