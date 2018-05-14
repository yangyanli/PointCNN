#!/usr/bin/python3
"""Calculate IoU of part segmentation task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_gt', '-g', help='Path to ground truth folder', required=True)
    parser.add_argument('--folder_pred', '-p', help='Path to prediction folder', required=True)
    parser.add_argument('--folder_data', '-d', help='Path to point cloud data folder')
    parser.add_argument('--part_avg', '-a', action='store_true', help='Use part level average')
    args = parser.parse_args()
    print(args)

    category_id_to_name = {
        2691156: 'Airplane',
        2773838: 'Bag',
        2954340: 'Cap',
        2958343: 'Car',
        3001627: 'Chair',
        3261776: 'Earphone',
        3467517: 'Guitar',
        3624134: 'Knife',
        3636649: 'Lamp',
        3642806: 'Laptop',
        3790512: 'Motorbike',
        3797390: 'Mug',
        3948459: 'Pistol',
        4099429: 'Rocket',
        4225987: 'Skateboard',
        4379243: 'Table'}

    categories = sorted(os.listdir(args.folder_gt))

    label_min = sys.maxsize
    for category in categories:
        category_folder_gt = os.path.join(args.folder_gt, category)
        filenames = sorted(os.listdir(category_folder_gt))
        for filename in filenames:
            filepath_gt = os.path.join(category_folder_gt, filename)
            label_gt = np.loadtxt(filepath_gt).astype(np.int32)
            label_min = min(label_min, np.amin(label_gt))

    IoU = 0.0
    total_num = 0
    for category in categories:
        category_folder_gt = os.path.join(args.folder_gt, category)
        category_folder_pred = os.path.join(args.folder_pred, category)
        if args.folder_data:
            category_folder_data = os.path.join(args.folder_data, category)
            category_folder_err = os.path.join(args.folder_pred+'_err_ply', category)

        IoU_category = 0.0
        filenames = sorted(os.listdir(category_folder_gt))
        for filename in filenames:
            filepath_gt = os.path.join(category_folder_gt, filename)
            filepath_pred = os.path.join(category_folder_pred, filename)
            label_gt = np.loadtxt(filepath_gt).astype(np.int32) - label_min
            label_pred = np.loadtxt(filepath_pred).astype(np.int32)

            if args.folder_data:
                filepath_data = os.path.join(category_folder_data, filename[:-3]+'pts')
                filepath_err = os.path.join(category_folder_err, filename[:-3] + 'ply')
                coordinates = [[float(value) for value in xyz.split(' ')]
                               for xyz in open(filepath_data, 'r') if len(xyz.split(' ')) == 3]
                assert (label_gt.shape[0] == len(coordinates))
                data_utils.save_ply_property(np.array(coordinates), (label_gt == label_pred), 6, filepath_err)

            if args.part_avg:
                label_max = np.amax(label_gt)
                IoU_part = 0.0
                for label_idx in range(label_max+1):
                    locations_gt = (label_gt == label_idx)
                    locations_pred = (label_pred == label_idx)
                    I_locations = np.logical_and(locations_gt, locations_pred)
                    U_locations = np.logical_or(locations_gt, locations_pred)
                    I = np.sum(I_locations) + np.finfo(np.float32).eps
                    U = np.sum(U_locations) + np.finfo(np.float32).eps
                    IoU_part =  IoU_part + I/U
                IoU_sample = IoU_part / (label_max+1)
            else:
                label_correct_locations = (label_gt == label_pred)
                IoU_sample = np.sum(label_correct_locations) / label_gt.size
            IoU_category = IoU_category + IoU_sample
        IoU = IoU + IoU_category
        IoU_category = IoU_category / len(filenames)
        if category.isdigit():
            print("IoU of %s: " % (category_id_to_name[int(category)]), IoU_category)
        else:
            print("IoU of %s: " % category, IoU_category)
        total_num = total_num + len(filenames)
    IoU = IoU / total_num
    print("IoU: ", IoU)


if __name__ == '__main__':
    main()
