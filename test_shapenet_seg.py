#!/usr/bin/python3
"""Testing On ShapeNet Parts Segmentation Task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import argparse
import importlib
import data_utils
import numpy as np
import tensorflow as tf
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', '-f', help='Path to input .h5 filelist (.txt)', required=True)
    parser.add_argument('--category', '-c', help='Path to category list file (.txt)', required=True)
    parser.add_argument('--data_folder', '-d', help='Path to *.pts directory', required=True)
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load', required=True)
    parser.add_argument('--repeat_num', '-r', help='Repeat number', type=int, default=1)
    parser.add_argument('--sample_num', help='Point sample num', type=int, default=2048)
    parser.add_argument('--model', '-m', help='Model to use', required=True)
    parser.add_argument('--setting', '-x', help='Setting to use', required=True)
    parser.add_argument('--save_ply', '-s', help='Save results as ply', action='store_true')
    args = parser.parse_args()
    print(args)

    model = importlib.import_module(args.model)
    setting_path = os.path.join(os.path.dirname(__file__), args.model)
    sys.path.append(setting_path)
    setting = importlib.import_module(args.setting)

    sample_num = setting.sample_num

    output_folder = args.data_folder + '_pred_nips_' + str(args.repeat_num)
    category_list = [(category, int(label_num)) for (category, label_num) in
                     [line.split() for line in open(args.category, 'r')]]
    offset = 0
    category_range = dict()
    for category, category_label_seg_max in category_list:
        category_range[category] = (offset, offset + category_label_seg_max)
        offset = offset + category_label_seg_max
        folder = os.path.join(output_folder, category)
        if not os.path.exists(folder):
            os.makedirs(folder)

    input_filelist = []
    output_filelist = []
    output_ply_filelist = []
    for category in sorted(os.listdir(args.data_folder)):
        data_category_folder = os.path.join(args.data_folder, category)
        for filename in sorted(os.listdir(data_category_folder)):
            input_filelist.append(os.path.join(args.data_folder, category, filename))
            output_filelist.append(os.path.join(output_folder, category, filename[0:-3] + 'seg'))
            output_ply_filelist.append(os.path.join(output_folder + '_ply', category, filename[0:-3] + 'ply'))

    # Prepare inputs
    print('{}-Preparing datasets...'.format(datetime.now()))
    data, label, data_num, _, _ = data_utils.load_seg(args.filelist)

    batch_num = data.shape[0]
    max_point_num = data.shape[1]
    batch_size = args.repeat_num * math.ceil(data.shape[1] / sample_num)

    print('{}-{:d} testing batches.'.format(datetime.now(), batch_num))

    ######################################################################
    # Placeholders
    indices = tf.placeholder(tf.int32, shape=(batch_size, None, 2), name="indices")
    is_training = tf.placeholder(tf.bool, name='is_training')
    pts_fts = tf.placeholder(tf.float32, shape=(None, max_point_num, setting.data_dim), name='pts_fts')
    ######################################################################

    ######################################################################
    pts_fts_sampled = tf.gather_nd(pts_fts, indices=indices, name='pts_fts_sampled')
    if setting.data_dim > 3:
        points_sampled, features_sampled = tf.split(pts_fts_sampled,
                                                    [3, setting.data_dim - 3],
                                                    axis=-1,
                                                    name='split_points_features')
        if not setting.use_extra_features:
            features_sampled = None
    else:
        points_sampled = pts_fts_sampled
        features_sampled = None

    net = model.Net(points_sampled, features_sampled, is_training, setting)
    logits = net.logits
    probs_op = tf.nn.softmax(logits, name='probs')

    saver = tf.train.Saver()

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

    with tf.Session() as sess:
        # Load the model
        saver.restore(sess, args.load_ckpt)
        print('{}-Checkpoint loaded from {}!'.format(datetime.now(), args.load_ckpt))

        indices_batch_indices = np.tile(np.reshape(np.arange(batch_size), (batch_size, 1, 1)), (1, sample_num, 1))
        for batch_idx in range(batch_num):
            points_batch = data[[batch_idx] * batch_size, ...]
            object_label = label[batch_idx]
            point_num = data_num[batch_idx]
            category = category_list[object_label][0]
            label_start, label_end = category_range[category]

            tile_num = math.ceil((sample_num * batch_size) / point_num)
            indices_shuffle = np.tile(np.arange(point_num), tile_num)[0:sample_num * batch_size]
            np.random.shuffle(indices_shuffle)
            indices_batch_shuffle = np.reshape(indices_shuffle, (batch_size, sample_num, 1))
            indices_batch = np.concatenate((indices_batch_indices, indices_batch_shuffle), axis=2)

            probs = sess.run([probs_op],
                                feed_dict={
                                    pts_fts: points_batch,
                                    indices: indices_batch,
                                    is_training: False,
                                })
            probs_2d = np.reshape(probs, (sample_num * batch_size, -1))
            predictions = [(-1, 0.0)] * point_num
            for idx in range(sample_num * batch_size):
                point_idx = indices_shuffle[idx]
                probs = probs_2d[idx, label_start:label_end]
                confidence = np.amax(probs)
                seg_idx = np.argmax(probs)
                if confidence > predictions[point_idx][1]:
                    predictions[point_idx] = (seg_idx, confidence)

            labels = []
            with open(output_filelist[batch_idx], 'w') as file_seg:
                for seg_idx, _ in predictions:
                    file_seg.write('%d\n' % (seg_idx))
                    labels.append(seg_idx)

            # read the coordinates from the txt file for verification
            coordinates = [[float(value) for value in xyz.split(' ')]
                           for xyz in open(input_filelist[batch_idx], 'r') if len(xyz.split(' ')) == 3]
            assert (point_num == len(coordinates))
            if args.save_ply:
                data_utils.save_ply_property(np.array(coordinates), np.array(labels), 6, output_ply_filelist[batch_idx])

            print('{}-[Testing]-Iter: {:06d} saved to {}'.format(datetime.now(), batch_idx, output_filelist[batch_idx]))
            sys.stdout.flush()
            ######################################################################
        print('{}-Done!'.format(datetime.now()))


if __name__ == '__main__':
    main()
