#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os


gt_label_filenames = []
pred_label_filenames = []
DATA_DIR = os.listdir("../../data/S3DIS")

for area in DATA_DIR:
    Rooms = os.listdir(os.path.join("../../data/S3DIS",area))
    for room in Rooms:
        path_gt_label = os.path.join("../../data/S3DIS",area,room,'label.npy')
        path_pred_label = os.path.join("../../data/S3DIS",area,room,'pred.npy')
        pred_label_filenames.append(path_pred_label)
        gt_label_filenames.append(path_gt_label)


num_room = len(gt_label_filenames)

#pred_data_label_filenames = gt_label_filenames
print(num_room)
print(len(pred_label_filenames))
assert(num_room == len(pred_label_filenames))

gt_classes = [0 for _ in range(13)]
positive_classes = [0 for _ in range(13)]
true_positive_classes = [0 for _ in range(13)]


for i in range(num_room):
    print(i,"/"+str(num_room))
    print(pred_label_filenames[i])
    pred_label = np.loadtxt(pred_label_filenames[i])
    gt_label = np.load(gt_label_filenames[i])
    for j in range(gt_label.shape[0]):
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
