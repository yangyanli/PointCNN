#!/usr/bin/python3
'''Convert MNIST to points.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import h5py
import random
import argparse
import numpy as np
from mnist import MNIST
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', help='Path to data folder')
    parser.add_argument('--point_num', '-p', help='Point number for each sample', type=int, default=256)
    parser.add_argument('--save_ply', '-s', help='Convert .pts to .ply', action='store_true')
    args = parser.parse_args()
    print(args)

    batch_size = 2048

    folder_mnist = args.folder if args.folder else '../../data/mnist/zips'
    folder_pts = os.path.join(os.path.dirname(folder_mnist), 'pts')

    mnist_data = MNIST(folder_mnist)
    mnist_train_test = [(mnist_data.load_training(), 'train'), (mnist_data.load_testing(), 'test')]

    data = np.zeros((batch_size, args.point_num, 4))
    label = np.zeros((batch_size), dtype=np.int32)
    for ((images, labels), tag) in mnist_train_test:
        idx_h5 = 0
        filename_filelist_h5 = os.path.join(os.path.dirname(folder_mnist), '%s_files.txt' % tag)
        point_num_total = 0
        with open(filename_filelist_h5, 'w') as filelist_h5:
            for idx_img, image in enumerate(images):
                points = []
                pixels = []
                for idx_pixel, pixel in enumerate(image):
                    if pixel == 0:
                        continue
                    x = idx_pixel // 28
                    z = idx_pixel % 28
                    points.append((x, random.random() * 1e-6, z))
                    pixels.append(pixel)
                point_num_total = point_num_total + len(points)
                pixels_sum = sum(pixels)
                probs = [pixel / pixels_sum for pixel in pixels]
                indices = np.random.choice(list(range(len(points))), size=args.point_num,
                                           replace=(len(points) < args.point_num), p=probs)
                points_array = np.array(points)[indices]
                pixels_array_1d = (np.array(pixels)[indices].astype(np.float32) / 255) - 0.5
                pixels_array = np.expand_dims(pixels_array_1d, axis=-1)

                points_min = np.amin(points_array, axis=0)
                points_max = np.amax(points_array, axis=0)
                points_center = (points_min + points_max) / 2
                scale = np.amax(points_max - points_min) / 2
                points_array = (points_array - points_center) * (0.8 / scale)

                if args.save_ply:
                    filename_pts = os.path.join(folder_pts, tag, '{:06d}.ply'.format(idx_img))
                    data_utils.save_ply(points_array, filename_pts, colors=np.tile(pixels_array, (1, 3)) + 0.5)

                idx_in_batch = idx_img % batch_size
                data[idx_in_batch, ...] = np.concatenate((points_array, pixels_array), axis=-1)
                label[idx_in_batch] = labels[idx_img]
                if ((idx_img + 1) % batch_size == 0) or idx_img == len(images) - 1:
                    item_num = idx_in_batch + 1
                    filename_h5 = os.path.join(os.path.dirname(folder_mnist), '%s_%d.h5' % (tag, idx_h5))
                    print('{}-Saving {}...'.format(datetime.now(), filename_h5))
                    filelist_h5.write('./%s_%d.h5\n' % (tag, idx_h5))

                    file = h5py.File(filename_h5, 'w')
                    file.create_dataset('data', data=data[0:item_num, ...])
                    file.create_dataset('label', data=label[0:item_num, ...])
                    file.close()

                    idx_h5 = idx_h5 + 1
        print('Average point number in each sample is : %f!' % (point_num_total / len(images)))


if __name__ == '__main__':
    main()
