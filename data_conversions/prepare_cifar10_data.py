#!/usr/bin/python3
'''Convert CIFAR-10 to points.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import h5py
import random
import tarfile
import argparse
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', help='Path to data folder')
    parser.add_argument('--save_ply', '-s', help='Convert .pts to .ply', action='store_true')
    args = parser.parse_args()
    print(args)

    batch_size = 2048

    folder_cifar10 = args.folder if args.folder else '../../data/cifar10/cifar-10-batches-py'
    folder_pts = os.path.join(os.path.dirname(folder_cifar10), 'pts')

    train_test_files = [('train', ['data_batch_%d' % (idx + 1) for idx in range(5)]),
                        ('test', ['test_batch'])]

    data = np.zeros((batch_size, 1024, 6))
    label = np.zeros((batch_size), dtype=np.int32)
    for tag, filelist in train_test_files:
        data_list = []
        labels_list = []
        for filename in filelist:
            batch = unpickle(os.path.join(folder_cifar10, filename))
            data_list.append(np.reshape(batch[b'data'], (10000, 3, 32, 32)))
            labels_list.append(batch[b'labels'])
        images = np.concatenate(data_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        idx_h5 = 0
        filename_filelist_h5 = os.path.join(os.path.dirname(folder_cifar10), '%s_files.txt' % tag)
        with open(filename_filelist_h5, 'w') as filelist_h5:
            for idx_img, image in enumerate(images):
                points = []
                pixels = []
                for x in range(32):
                    for z in range(32):
                        points.append((x, random.random() * 1e-6, z))
                        pixels.append((image[0, x, z], image[1, x, z], image[2, x, z]))
                points_array = np.array(points)
                pixels_array = (np.array(pixels).astype(np.float32) / 255)-0.5

                points_min = np.amin(points_array, axis=0)
                points_max = np.amax(points_array, axis=0)
                points_center = (points_min + points_max) / 2
                scale = np.amax(points_max - points_min) / 2
                points_array = (points_array - points_center) * (0.8 / scale)

                if args.save_ply:
                    filename_pts = os.path.join(folder_pts, tag, '{:06d}.ply'.format(idx_img))
                    data_utils.save_ply(points_array, filename_pts, colors=pixels_array+0.5)

                idx_in_batch = idx_img % batch_size
                data[idx_in_batch, ...] = np.concatenate((points_array, pixels_array), axis=-1)
                label[idx_in_batch] = labels[idx_img]
                if ((idx_img + 1) % batch_size == 0) or idx_img == len(images) - 1:
                    item_num = idx_in_batch + 1
                    filename_h5 = os.path.join(os.path.dirname(folder_cifar10), '%s_%d.h5' % (tag, idx_h5))
                    print('{}-Saving {}...'.format(datetime.now(), filename_h5))
                    filelist_h5.write('./%s_%d.h5\n' % (tag, idx_h5))

                    file = h5py.File(filename_h5, 'w')
                    file.create_dataset('data', data=data[0:item_num, ...])
                    file.create_dataset('label', data=label[0:item_num, ...])
                    file.close()

                    idx_h5 = idx_h5 + 1

if __name__ == '__main__':
    main()
