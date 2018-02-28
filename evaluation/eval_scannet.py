#!/usr/bin/python3
"""Merge blocks and evaluate scannet"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import plyfile
import numpy as np


def dir(root, type='f', addroot=True):
    dirList = []
    fileList = []
    files = os.listdir(root)
    for f in files:
        if (os.path.isdir(root + f)):
            if addroot == True:
                dirList.append(root + f)
            else:
                dirList.append(f)
        if (os.path.isfile(root + f)):
            if addroot == True:
                fileList.append(root + f)
            else:
                fileList.append(f)
    if type == "f":
        fileList.sort()
        return fileList
    elif type == "d":
        dirList.sort()
        return dirList
    else:
        print("ERROR: TMC.dir(root,type) type must be [f] for file or [d] for dir")

        return 0


block_data_root = "./scannet_split_dataset/val_data/01/"
block_label_root = "./scannet_split_dataset/val_label/01/"
block_trans_root = "./scannet_split_dataset/val_trans/01/"
block_pred_root = "./_pred_RGB4/01/"

save_voxel_re = True

seg_pred_out_root = "./out/seg_pred_voxel/"
seg_label_out_root = "./out/seg_label_voxel/"
pts_out_root = "./out/pts_voxel/"

# check the path
if not os.path.exists(seg_pred_out_root):
    print(seg_pred_out_root, "Not Exists! Create", seg_pred_out_root)
    os.makedirs(seg_pred_out_root)
if not os.path.exists(seg_label_out_root):
    print(seg_label_out_root, "Not Exists! Create", seg_label_out_root)
    os.makedirs(seg_label_out_root)
if not os.path.exists(pts_out_root):
    print(pts_out_root, "Not Exists! Create", pts_out_root)
    os.makedirs(pts_out_root)

block_pred_list = dir(block_pred_root)

scene_list = {}
pts_acc_list = []
vox_acc_list = []

# get scene-block dict
for block_pred_file in block_pred_list:

    name = block_pred_file.split("/")[-1].split(".")[0]
    scene_name = name[0:5]
    block_name = name[5:8]

    if scene_name not in scene_list.keys():
        scene_list[scene_name] = []

    scene_list[scene_name].append(block_name)

for pk, scene in enumerate(scene_list.keys()):

    print("process scene", scene, "(" + str(pk) + "/" + str(len(scene_list.keys())) + ")")

    seg_pred_outfile = seg_pred_out_root + scene + ".seg"
    seg_label_outfile = seg_label_out_root + scene + ".seg"
    pts_out_file = pts_out_root + scene + ".pts"

    block_trans_list = {}
    merge_pts = []
    merge_label_seg = []
    merge_pred_seg = []

    # read trans
    block_trans_file = block_trans_root + scene + ".trs"

    with open(block_trans_file, "r") as trs_f:

        for line in trs_f:
            line_s = line.strip().split(" ")
            block_trans_list[line_s[0]] = []
            block_trans_list[line_s[0]] = [float(line_s[1]), float(line_s[2]), float(line_s[3])]

    for block in scene_list[scene]:

        block_pred_file = block_pred_root + scene + block + ".seg"
        block_label_file = block_label_root + scene + block + ".seg"
        block_pts_file = block_data_root + scene + block + ".pts"
        block_trans = block_trans_list[scene + block]

        with open(block_pts_file, "r") as pts_f:

            for line in pts_f:
                line_s = line.strip().split(" ")
                pt = [float(line_s[0]) + block_trans[0], float(line_s[1]) + block_trans[2],
                      float(line_s[2]) + block_trans[1]]
                merge_pts.append(pt)

        with open(block_label_file, "r") as seg_f:

            for line in seg_f:
                line_s = line.strip()
                merge_label_seg.append(int(line_s))

        with open(block_pred_file, "r") as seg_p_f:

            for line in seg_p_f:
                line_s = line.strip().split(" ")
                seg = int(line_s[0])
                merge_pred_seg.append(seg)

    # compute scene pts accuracy
    c_accpt = 0
    c_ignore = 0

    # ignore label 0 (scannet unannotated)
    for k, label in enumerate(merge_label_seg):

        if label == 0:
            c_ignore = c_ignore + 1

        elif label == merge_pred_seg[k]:
            c_accpt = c_accpt + 1

    acc = c_accpt * 1.0 / (len(merge_label_seg) - c_ignore)
    pts_acc_list.append([c_accpt, (len(merge_label_seg) - c_ignore)])
    print("\npts acc:", acc)

    # compute scene voxel accuracy (follow scannet and pointnet++)

    # follow pointnet++ voxel size
    res = 0.0484
    coordmax = np.max(merge_pts, axis=0)
    coordmin = np.min(merge_pts, axis=0)
    nvox = np.ceil((coordmax - coordmin) / res)
    vidx = np.ceil((merge_pts - coordmin) / res)
    vidx = vidx[:, 0] + vidx[:, 1] * nvox[0] + vidx[:, 2] * nvox[0] * nvox[1]
    uvidx, vpidx = np.unique(vidx, return_index=True)

    # compute voxel label
    uvlabel = np.array(merge_label_seg)[vpidx]

    # compute voxel pred (follow pointnet++ majority voting)
    uvpred_tp = []
    label_pred_dict = {}

    for uidx in uvidx:
        label_pred_dict[int(uidx)] = []
    for k, pred_seg in enumerate(merge_pred_seg):
        label_pred_dict[int(vidx[k])].append(pred_seg)
    for uidx in uvidx:
        uvpred_tp.append(np.argmax(np.bincount(label_pred_dict[int(uidx)])))

    # compute accuracy
    c_accvox = 0
    c_ignore = 0

    # ignore label 0 (scannet unannotated)
    for k, label in enumerate(uvlabel):

        if label == 0:
            c_ignore = c_ignore + 1

        elif label == uvpred_tp[k]:
            c_accvox = c_accvox + 1

    acc = c_accvox * 1.0 / (len(uvlabel) - c_ignore)
    vox_acc_list.append([c_accvox, (len(uvlabel) - c_ignore)])
    print("voxel acc", acc)

    # compute cls accuracy
    label_unique, label_count = np.unique(uvlabel, return_counts=True)
    print("\nunique label:", label_unique, "count", label_count)

    acc_dict = {}
    for k, label in enumerate(uvlabel):
        if label not in acc_dict.keys():
            acc_dict[label] = 0
        if label == uvpred_tp[k]:
            acc_dict[label] = acc_dict[label] + 1

    print("acc dict:", acc_dict)

    for k, ul in enumerate(label_unique):
        if ul == 0:
            print("label:", ul, "acc:", acc_dict[ul] * 1.0 / label_count[k], "(label 0 is scannet unannotated, ignore)")
        else:
            print("label:", ul, "acc:", acc_dict[ul] * 1.0 / label_count[k])

    # compute avg accuracy
    acc_sum = 0
    count_all = 0

    for acc in pts_acc_list:
        acc_sum = acc_sum + acc[0]
        count_all = count_all + acc[1]

    print("\npts avg acc", "(" + str(pk) + "/" + str(len(scene_list.keys())) + ")", acc_sum * 1.0 / count_all)

    acc_sum = 0
    count_all = 0

    for acc in vox_acc_list:
        acc_sum = acc_sum + acc[0]
        count_all = count_all + acc[1]

    print("voxel avg acc", "(" + str(pk) + "/" + str(len(scene_list.keys())) + ")", acc_sum * 1.0 / count_all, '\n')

    if save_voxel_re:

        voxel_label_tabel = {}
        save_pts = []
        save_seg = []
        save_pred = []

        for k, label in enumerate(uvlabel):
            voxel_label_tabel[int(uvidx[k])] = [label, uvpred_tp[k]]

        for k, vid in enumerate(vidx):
            save_pts.append(merge_pts[k])
            label_voxel = voxel_label_tabel[int(vid)]
            save_seg.append(label_voxel[0])
            save_pred.append(label_voxel[1])

        # savepts
        with open(pts_out_file, "w") as f:
            for pt in save_pts:
                f.writelines(str(pt[0]) + " " + str(pt[1]) + " " + str(pt[2]) + "\n")

        # saveseg
        with open(seg_label_outfile, "w") as f:
            for s in save_seg:
                f.writelines(str(s) + "\n")

        # savesegpred
        with open(seg_pred_outfile, "w") as f:
            for s in save_pred:
                f.writelines(str(s) + "\n")