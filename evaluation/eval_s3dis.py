#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np


gt_label_filenames = []
pred_label_filenames = []

DEFAULT_DATA_DIR = '../../data/s3dis'
NUM_CLASSES = 13

p = argparse.ArgumentParser()
p.add_argument(
    "-d", "--data", dest='data_dir',
    default=DEFAULT_DATA_DIR,
    help="Path to S3DIS data (default is %s)" % DEFAULT_DATA_DIR)

args = p.parse_args()

for area in os.listdir(args.data_dir):
    path_area = os.path.join(args.data_dir, area)
    if not os.path.isdir(path_area):
        continue
    Rooms = os.listdir(path_area)
    for room in Rooms:
        path_room = os.path.join(path_area, room)
        if not os.path.isdir(path_room):
            continue
        path_gt_label = os.path.join(path_room, 'label.npy')
        if not os.path.exists(path_gt_label):
            print("%s does not exist, skipping" % path_gt_label)
            continue
        path_pred_label = os.path.join(path_room, 'pred.npy')
        if not os.path.exists(path_pred_label):
            print("%s does not exist, skipping" % path_pred_label)
            continue
        pred_label_filenames.append(path_pred_label)
        gt_label_filenames.append(path_gt_label)

num_room = len(gt_label_filenames)
num_preds = len(pred_label_filenames)
assert num_room == num_preds

print("Found {} predictions".format(num_room))

gt_classes = [0] * NUM_CLASSES
positive_classes = [0] * NUM_CLASSES
true_positive_classes = [0] * NUM_CLASSES

print("Evaluating predictions:")
for i in range(num_room):
    print("  {} ({}/{})".format(pred_label_filenames[i], i + 1, num_room))
    pred_label = np.loadtxt(pred_label_filenames[i])
    gt_label = np.load(gt_label_filenames[i])
    for j in range(gt_label.shape[0]):
        gt_l = int(gt_label[j])
        pred_l = int(pred_label[j])
        gt_classes[gt_l] += 1
        positive_classes[pred_l] += 1
        true_positive_classes[gt_l] += int(gt_l==pred_l)

print("Classes:\t{}".format("\t".join(map(str, gt_classes))))
print("Positive:\t{}".format("\t".join(map(str, positive_classes))))
print("True positive:\t{}".format("\t".join(map(str, true_positive_classes))))
print("Overall accuracy: {0}".format(sum(true_positive_classes)/float(sum(positive_classes))))

print("Class IoU:")
iou_list = []
for i in range(13):
    iou = true_positive_classes[i]/float(gt_classes[i]+positive_classes[i]-true_positive_classes[i])
    print("  {}: {}".format(i, iou))
    iou_list.append(iou)

print("Average IoU: {}".format(sum(iou_list)/13.0))
