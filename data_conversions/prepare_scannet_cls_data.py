#!/usr/bin/python3
'''Convert ScanNet pts to h5.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import argparse
import numpy as np
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', help='Path to data folder')
    args = parser.parse_args()
    print(args)

    batch_size = 2048
    sample_num = 2048

    folder_scanenet = args.folder if args.folder else '../../data/scannet/cls'
    train_test_folders = ['train', 'test']

    label_list = []
    for folder in train_test_folders:
        folder_pts = os.path.join(folder_scanenet, folder, 'pts')
        for filename in os.listdir(folder_pts):
            label_list.append(int(filename[:-4].split('_')[-1]))
    label_list = sorted(set(label_list))
    print('label_num:', len(label_list))
    label_dict = dict()
    for idx, label in enumerate(label_list):
        label_dict[label] = idx

    data = np.zeros((batch_size, sample_num, 6))
    label = np.zeros((batch_size), dtype=np.int32)
    for folder in train_test_folders:
        folder_pts = os.path.join(folder_scanenet, folder, 'pts')

        idx_h5 = 0
        filename_filelist_h5 = os.path.join(folder_scanenet, '%s_files.txt' % folder)
        with open(filename_filelist_h5, 'w') as filelist_h5:
            filelist = os.listdir(folder_pts)
            for idx_pts, filename in enumerate(filelist):
                label_object = label_dict[int(filename[:-4].split('_')[-1])]
                filename_pts = os.path.join(folder_pts, filename)
                xyzrgbs = np.array([[float(value) for value in xyzrgb.split(' ')]
                               for xyzrgb in open(filename_pts, 'r') if len(xyzrgb.split(' ')) == 6])
                np.random.shuffle(xyzrgbs)
                pt_num = xyzrgbs.shape[0]
                indices = np.random.choice(pt_num, sample_num, replace=(pt_num < sample_num))
                points_array = xyzrgbs[indices]
                points_array[..., 3:] = points_array[..., 3:]/255 - 0.5 # normalize colors

                idx_in_batch = idx_pts % batch_size
                data[idx_in_batch, ...] = points_array
                label[idx_in_batch] = label_object
                if ((idx_pts + 1) % batch_size == 0) or idx_pts == len(filelist) - 1:
                    item_num = idx_in_batch + 1
                    filename_h5 = os.path.join(folder_scanenet, '%s_%d.h5' % (folder, idx_h5))
                    print('{}-Saving {}...'.format(datetime.now(), filename_h5))
                    filelist_h5.write('./%s_%d.h5\n' % (folder, idx_h5))

                    file = h5py.File(filename_h5, 'w')
                    file.create_dataset('data', data=data[0:item_num, ...])
                    file.create_dataset('label', data=label[0:item_num, ...])
                    file.close()

                    idx_h5 = idx_h5 + 1

if __name__ == '__main__':
    main()
