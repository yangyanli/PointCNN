#!/usr/bin/python3
"""Testing On Segmentation Task."""

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
    parser.add_argument('--filelist', '-t', help='Path to input .h5 filelist (.txt)', required=True)
    parser.add_argument('--data_folder', '-f', help='Path to *.pts directory', required=True)
    parser.add_argument('--category', '-c', help='Path to category list file (.txt)', required=True)
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load', required=True)
    parser.add_argument('--repeat_num', '-r', help='Repeat number', type=int, default=1)
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
    num_parts = setting.num_parts

    # Prepare output folder
    output_folder = args.data_folder + '_pred_RGB'+str(args.repeat_num)
    category_list = [(category, int(label_num)) for (category, label_num) in
                     [line.split() for line in open(args.category, 'r')]]
    for category, _ in category_list:
        folder = os.path.join(output_folder, category)
        if not os.path.exists(folder):
            os.makedirs(folder)

    # prepare input pts path, output seg path, output ply path 
    input_filelist = []
    output_filelist = []
    output_ply_filelist = []
    for category in sorted(os.listdir(args.data_folder)):
        data_category_folder = os.path.join(args.data_folder, category)
        for filename in sorted(os.listdir(data_category_folder)):
            input_filelist.append(os.path.join(args.data_folder, category, filename))
            output_filelist.append(os.path.join(output_folder, category, filename[0:-3]+'seg'))
            output_ply_filelist.append(os.path.join(output_folder+'_ply', category, filename[0:-3] + 'ply'))

    # Prepare inputs
    print('{}-Preparing datasets...'.format(datetime.now()))
    data, _, data_num, _ = data_utils.load_seg(args.filelist)

    batch_num = data.shape[0]
    #point_num
    max_point_num = data.shape[1]
    batch_size = args.repeat_num*math.ceil(data.shape[1]/sample_num)

    print('{}-{:d} testing batches.'.format(datetime.now(), batch_num))

    ######################################################################
    # Placeholders
    indices = tf.placeholder(tf.int32, shape=(batch_size, None, 2), name="indices")
    is_training = tf.placeholder(tf.bool, name='is_training')
    pts_fts = tf.placeholder(tf.float32, shape=(batch_size, max_point_num, setting.data_dim), name='points')

    ######################################################################
    
    features_augmented = None
    if setting.data_dim > 3:
        points, features = tf.split(pts_fts, [3, setting.data_dim - 3], axis=-1, name='split_points_features')
        if setting.use_extra_features:
            features_augmented = tf.gather_nd(features, indices=indices, name='features_sampled')
    else:
        points = pts_fts

    points_sampled = tf.gather_nd(points, indices=indices, name='points_sampled')

    net = model.Net(points_sampled, features_augmented, num_parts, is_training, setting)
    _, seg_probs_op = net.logits, net.probs

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    #for restore model
    saver = tf.train.Saver()

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

    with tf.Session() as sess:
        # Load the model
        saver.restore(sess, args.load_ckpt)
        print('{}-Checkpoint loaded from {}!'.format(datetime.now(), args.load_ckpt))

        indices_batch_indices = np.tile(np.reshape(np.arange(batch_size), (batch_size, 1, 1)), (1, sample_num, 1))

        for batch_idx in range(batch_num):

            points_batch = data[[batch_idx]*batch_size, ...]
            point_num = data_num[batch_idx]

            coordinates = [[float(value) for value in xyz.split(' ')]
                           for xyz in open(input_filelist[batch_idx], 'r') if len(xyz.split(' ')) == 6]
            assert(point_num == len(coordinates))

            tile_num = math.ceil((sample_num*batch_size)/point_num)
            indices_shuffle = np.tile(np.arange(point_num), tile_num)[0:sample_num*batch_size]
            np.random.shuffle(indices_shuffle)
            indices_batch_shuffle = np.reshape(indices_shuffle, (batch_size, sample_num, 1))
            indices_batch = np.concatenate((indices_batch_indices, indices_batch_shuffle), axis=2)

            _, seg_probs = \
                sess.run([update_ops, seg_probs_op],
                         feed_dict={
                             pts_fts: points_batch,
                             indices: indices_batch,
                             is_training: False,
                         })


            seg_probs_2d = np.reshape(seg_probs, (sample_num*batch_size, -1))
            
            predictions = [(-1, 0.0, [])]*point_num
            
            for idx in range(sample_num*batch_size):
                point_idx = indices_shuffle[idx]
                point_seg_probs = seg_probs_2d[idx, :]
                prob = np.amax(point_seg_probs)
                seg_idx = np.argmax(point_seg_probs)
                if prob > predictions[point_idx][1]:

                    predictions[point_idx] = [seg_idx, prob, point_seg_probs]

            labels = []
            with open(output_filelist[batch_idx], 'w') as file_seg:
                for seg_idx, prob, probs in predictions:

                    file_seg.write(str(int(seg_idx)))

                    file_seg.write("\n")

                    labels.append(seg_idx)

            if args.save_ply:
                data_utils.save_ply_property(np.array(coordinates), np.array(labels), 6, output_ply_filelist[batch_idx])

            print('{}-[Testing]-Iter: {:06d} saved to {}'.format(datetime.now(), batch_idx, output_filelist[batch_idx]))
            sys.stdout.flush()
            ######################################################################
        print('{}-Done!'.format(datetime.now()))


if __name__ == '__main__':
    main()
