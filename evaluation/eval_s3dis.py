#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

BASE_DIR = os.path.dirname(__file__)

gt_label_filenames = []
path_pred_label = []

ROOT_DIR = os.path.join(BASE_DIR,"prepare_label_rgb")
Areas = os.listdir(ROOT_DIR)
for area in Areas:
    Rooms = os.listdir(os.path.join(ROOT_DIR,area))
    for room in Rooms:
        path_room = os.path.join(ROOT_DIR,area,room)
        path_label = os.path.join(path_room,"label.npy")
        gt_label_filenames.append(path_label)

num_room = len(gt_label_filenames)
pred_data_label_filenames = gt_label_filenames

gt_classes = [0 for _ in range(13)]
positive_classes = [0 for _ in range(13)]
true_positive_classes = [0 for _ in range(13)]

for i in range(num_room):
    print(i)
    data_label = np.load(pred_data_label_filenames[i])
    print(pred_data_label_filenames[i])
    print(data_label.dtype)
    pred_label = data_label[:,-1]
    print(pred_label.dtype)
    gt_label = np.load(gt_label_filenames[i])
    print(gt_label.dtype)
    for j in xrange(gt_label.shape[0]):
        gt_l = int(gt_label[j])
        pred_l = int(pred_label[j])
        gt_classes[gt_l] += 1
        positive_classes[pred_l] += 1
        true_positive_classes[gt_l] += int(gt_l==pred_l)

print(gt_classes)
print(positive_classes)
print(true_positive_classes)

print('Overall accuracy: {0}'.format(sum(true_positive_classes)/float(sum(positive_classes))))

print('IoU:')
iou_list = []
for i in range(13):
    iou = true_positive_classes[i]/float(gt_classes[i]+positive_classes[i]-true_positive_classes[i])
    print(iou)
    iou_list.append(iou)

print(sum(iou_list)/13.0)