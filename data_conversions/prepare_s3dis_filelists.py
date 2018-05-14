#!/usr/bin/python3
'''Prepare Filelists for S3DIS Segmentation Task.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import random
import argparse
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', help='Path to data folder')
    parser.add_argument('--h5_num', '-d', help='Number of h5 files to be loaded each time', type=int, default=8)
    parser.add_argument('--repeat_num', '-r', help='Number of repeatly using each loaded h5 list', type=int, default=2)

    args = parser.parse_args()
    print(args)

    root = args.folder if args.folder else '../../data/s3dis/'

    area_h5s = [[] for _ in range(6)]
    for area_idx in range(1, 7):
        folder = os.path.join(root, 'Area_%d' % area_idx)
        datasets = [dataset for dataset in os.listdir(folder)]
        for dataset in datasets:
            folder_dataset = os.path.join(folder, dataset)
            filename_h5s = ['./Area_%d/%s/%s\n' % (area_idx, dataset, filename) for filename in
                            os.listdir(folder_dataset)
                            if filename.endswith('.h5')]
            area_h5s[area_idx - 1].extend(filename_h5s)

    for area_idx in range(1, 7):
        train_h5 = [filename for idx in range(6) if idx + 1 != area_idx for filename in area_h5s[idx]]
        random.shuffle(train_h5)
        train_list = os.path.join(root, 'train_files_for_val_on_Area_%d.txt' % area_idx)
        print('{}-Saving {}...'.format(datetime.now(), train_list))
        with open(train_list, 'w') as filelist:
            list_num = math.ceil(len(train_h5) / args.h5_num)
            for list_idx in range(list_num):
                train_val_list_i = os.path.join(root, 'filelists',
                                                'train_files_for_val_on_Area_%d_g_%d.txt' % (area_idx, list_idx))
                os.makedirs(os.path.dirname(train_val_list_i), exist_ok=True)
                with open(train_val_list_i, 'w') as filelist_i:
                    for h5_idx in range(args.h5_num):
                        filename_idx = list_idx * args.h5_num + h5_idx
                        if filename_idx > len(train_h5) - 1:
                            break
                        filename_h5 = train_h5[filename_idx]
                        filelist_i.write('../' + filename_h5)
                for repeat_idx in range(args.repeat_num):
                    filelist.write('./filelists/train_files_for_val_on_Area_%d_g_%d.txt\n' % (area_idx, list_idx))

        val_h5 = area_h5s[area_idx - 1]
        val_list = os.path.join(root, 'val_files_Area_%d.txt' % area_idx)
        print('{}-Saving {}...'.format(datetime.now(), val_list))
        with open(val_list, 'w') as filelist:
            for filename_h5 in val_h5:
                filelist.write(filename_h5)


if __name__ == '__main__':
    main()
