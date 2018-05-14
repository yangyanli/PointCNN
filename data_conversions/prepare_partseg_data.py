#!/usr/bin/python3
'''Prepare Data for ShapeNet Segmentation Task.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import h5py
import argparse
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', help='Path to data folder')
    parser.add_argument('--save_ply', '-s', help='Convert .pts to .ply', action='store_true')
    args = parser.parse_args()
    print(args)

    root = args.folder if args.folder else '../../data/shapenet_partseg'
    folders = [(os.path.join(root,'train_data'), os.path.join(root,'train_label')),
               (os.path.join(root,'val_data'), os.path.join(root,'val_label')),
               (os.path.join(root,'test_data'), os.path.join(root,'test_label'))]
    category_label_seg_max_dict = dict()
    max_point_num = 0
    label_seg_min = sys.maxsize
    for data_folder, label_folder in folders:
        if not os.path.exists(data_folder):
            continue
        for category in sorted(os.listdir(data_folder)):
            if category not in category_label_seg_max_dict:
                category_label_seg_max_dict[category] = 0
            data_category_folder = os.path.join(data_folder, category)
            category_label_seg_max = 0
            for filename in sorted(os.listdir(data_category_folder)):
                data_filepath = os.path.join(data_category_folder, filename)
                coordinates = [xyz for xyz in open(data_filepath, 'r') if len(xyz.split(' ')) == 3]
                max_point_num = max(max_point_num, len(coordinates))

                if label_folder is not None:
                    label_filepath = os.path.join(label_folder, category, filename[0:-3] + 'seg')
                    label_seg_this = np.loadtxt(label_filepath).astype(np.int32)
                    assert (len(coordinates) == len(label_seg_this))
                    category_label_seg_max = max(category_label_seg_max, max(label_seg_this))
                    label_seg_min = min(label_seg_min, min(label_seg_this))
            category_label_seg_max_dict[category] = max(category_label_seg_max_dict[category], category_label_seg_max)
    category_label_seg_max_list = [(key, category_label_seg_max_dict[key]) for key in
                                   sorted(category_label_seg_max_dict.keys())]

    category_label = dict()
    offset = 0
    category_offset = dict()
    label_seg_max = max([category_label_seg_max for _, category_label_seg_max in category_label_seg_max_list])
    with open(os.path.join(root, 'categories.txt'), 'w') as file_categories:
        for idx, (category, category_label_seg_max) in enumerate(category_label_seg_max_list):
            file_categories.write('%s %d\n' % (category, category_label_seg_max - label_seg_min + 1))
            category_label[category] = idx
            category_offset[category] = offset
            offset = offset + category_label_seg_max - label_seg_min + 1

    print('part_num:', offset)
    print('max_point_num:', max_point_num)
    print(category_label_seg_max_list)

    batch_size = 2048
    data = np.zeros((batch_size, max_point_num, 3))
    data_num = np.zeros((batch_size), dtype=np.int32)
    label = np.zeros((batch_size), dtype=np.int32)
    label_seg = np.zeros((batch_size, max_point_num), dtype=np.int32)
    for data_folder, label_folder in folders:
        if not os.path.exists(data_folder):
            continue
        data_folder_ply = data_folder + '_ply'
        file_num = 0
        for category in sorted(os.listdir(data_folder)):
            data_category_folder = os.path.join(data_folder, category)
            file_num = file_num + len(os.listdir(data_category_folder))
        idx_h5 = 0
        idx = 0

        save_path = '%s/%s' % (os.path.dirname(data_folder), os.path.basename(data_folder)[0:-5])
        filename_txt = '%s_files.txt' % (save_path)
        ply_filepath_list = []
        with open(filename_txt, 'w') as filelist:
            for category in sorted(os.listdir(data_folder)):
                data_category_folder = os.path.join(data_folder, category)
                for filename in sorted(os.listdir(data_category_folder)):
                    data_filepath = os.path.join(data_category_folder, filename)
                    coordinates = [[float(value) for value in xyz.split(' ')]
                                   for xyz in open(data_filepath, 'r') if len(xyz.split(' ')) == 3]
                    idx_in_batch = idx % batch_size
                    data[idx_in_batch, 0:len(coordinates), ...] = np.array(coordinates)
                    data_num[idx_in_batch] = len(coordinates)
                    label[idx_in_batch] = category_label[category]

                    if label_folder is not None:
                        label_filepath = os.path.join(label_folder, category, filename[0:-3] + 'seg')
                        label_seg_this = np.loadtxt(label_filepath).astype(np.int32) - label_seg_min
                        assert (len(coordinates) == label_seg_this.shape[0])
                        label_seg[idx_in_batch, 0:len(coordinates)] = label_seg_this + category_offset[category]

                    data_ply_filepath = os.path.join(data_folder_ply, category, filename[:-3] + 'ply')
                    ply_filepath_list.append(data_ply_filepath)

                    if ((idx + 1) % batch_size == 0) or idx == file_num - 1:
                        item_num = idx_in_batch + 1
                        filename_h5 = '%s_%d.h5' % (save_path, idx_h5)
                        print('{}-Saving {}...'.format(datetime.now(), filename_h5))
                        filelist.write('./%s_%d.h5\n' % (os.path.basename(data_folder)[0:-5], idx_h5))

                        file = h5py.File(filename_h5, 'w')
                        file.create_dataset('data', data=data[0:item_num, ...])
                        file.create_dataset('data_num', data=data_num[0:item_num, ...])
                        file.create_dataset('label', data=label[0:item_num, ...])
                        file.create_dataset('label_seg', data=label_seg[0:item_num, ...])
                        file.close()

                        if args.save_ply:
                            data_utils.save_ply_property_batch(data[0:item_num, ...], label_seg[0:item_num, ...],
                                                               ply_filepath_list, data_num[0:item_num, ...],
                                                               label_seg_max - label_seg_min)
                        ply_filepath_list = []
                        idx_h5 = idx_h5 + 1
                    idx = idx + 1


if __name__ == '__main__':
    main()
