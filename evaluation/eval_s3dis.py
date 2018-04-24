#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

BASE_DIR = os.path.dirname(__file__)

gt_label_filenames = []
pred_label_filenames = []

GT_DIR = os.path.join(BASE_DIR,"out_part/train_ori_seg/Area6_data/01/")
gt_Areas = os.listdir(GT_DIR)
for gt_area in gt_Areas:
    gt_Rooms = os.listdir(os.path.join(GT_DIR,gt_area,"01"))
    for gt_room in gt_Rooms:
        path_gt_label = os.path.join(GT_DIR,gt_area,"01",gt_room)
        gt_label_filenames.append(path_gt_label)

num_room = len(gt_label_filenames)

PRED_DIR = os.path.join(BASE_DIR,"upsampling/upsample_pred_A6/seg/")

pred_Areas = os.listdir(PRED_DIR)
for pred_area in pred_Areas:
    pred_Rooms = os.listdir(os.path.join(PRED_DIR,pred_area,"seg"))
    for pred_room in pred_Rooms:
        path_pred_label = os.path.join(PRED_DIR,pred_area,"seg",pred_room)
        pred_label_filenames.append(path_pred_label)

assert(num_room == len(pred_label_filenames))

gt_classes = [0 for _ in range(13)]
positive_classes = [0 for _ in range(13)]
true_positive_classes = [0 for _ in range(13)]


for i in range(num_room):
    print(i,"/"+str(num_room))
    pred_label = np.loadtxt(pred_label_filenames[i])
    gt_label = np.loadtxt(gt_label_filenames[i])
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
