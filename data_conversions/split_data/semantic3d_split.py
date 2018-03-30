#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import math
import numpy as np

BASE_DIR = "../../../data/semantic3d"
npy_file_test = os.path.join(BASE_DIR, 'test')
npy_file_val = os.path.join(BASE_DIR, 'val')
npy_file_train = os.path.join(BASE_DIR, 'train')

res = 0.10  # 10cm
max_b = 5.0
overlap = 2.5
block_min_pnum = 300
sample_flag = False
max_pts = 4096

# out path
train_data_root = "../../../data/semantic3d/out_part/train_data/"
train_label_root = "../../../data/semantic3d/out_part/train_label/"
train_trans_root = "../../../data/semantic3d/out_part/train_trans/"

val_data_root = "../../../data/semantic3d/out_part/val_data/"
val_label_root = "../../../data/semantic3d/out_part/val_label/"
val_trans_root = "../../../data/semantic3d/out_part/val_trans/"


def pc_getbbox(pc):
    x = []
    y = []
    z = []

    for pts in pc:
        x.append(pts[0])
        y.append(pts[1])
        z.append(pts[2])

    boundary = [min(x), max(x), min(y), max(y), min(z), max(z)]

    return boundary


def unpickle(npy_file, out_data, out_label, out_trans):
    path_categories = sorted(os.listdir(npy_file))
    txt_file = []
    for file_filter in path_categories:
        if file_filter.split(".", 1)[-1] == "txt":
            txt_file.append(file_filter)
        else:
            continue

    for category in txt_file:

        print(category[:-4])
        if not os.path.exists(out_data + category.split("_", 1)[0]):
            print(out_data, "Not Exists! Create", out_data)
            os.makedirs(out_data + category.split("_", 1)[0])
        if not os.path.exists(out_label + category.split("_", 1)[0]):
            print(out_label, "Not Exists! Create", out_label)
            os.makedirs(out_label + category.split("_", 1)[0])
        if not os.path.exists(out_trans + category.split("_", 1)[0]):
            print(out_trans, "Not Exists! Create", out_trans)
            os.makedirs(out_trans + category.split("_", 1)[0])

        path_data = os.path.join(npy_file, category)
        path_seg = os.path.join(npy_file, category[:-4] + ".labels")

        print("\nProcess", path_seg)

        pf = []
        temp_pf = []
        sf = []
        index_seg = 0
        list_index_seg = []

        with open(path_seg, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if int(line) != 0:
                    list_index_seg.append(int(index_seg))
                    sf.append(int(line))
                index_seg += 1

        print("load ",path_data)
        with open(path_data, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                temp_pf.append(line)

        for i in list_index_seg:
            temp = temp_pf[i].split()
            pf.append([float(temp[0]),
                       float(temp[1]),
                       float(temp[2]),
                       int(temp[4]),
                       int(temp[5]),
                       int(temp[6])])

        print("loaded data")
        pf = np.array(pf)
        sf = np.array(sf)

        # downsampling
        coordmax = np.max(pf, axis=0)
        coordmin = np.min(pf, axis=0)
        nvox = np.ceil((coordmax - coordmin) / res)
        vidx = np.ceil((pf - coordmin) / res)
        vidx = vidx[:, 0] + vidx[:, 1] * nvox[0] + vidx[:, 2] * nvox[0] * nvox[1]

        uvidx, vpidx = np.unique(vidx, return_index=True)
        # compute voxel label
        pf = np.array(pf)[vpidx].tolist()
        sf = np.array(sf)[vpidx].tolist()
        #########

        pts_num = len(pf)
        seg_num = len(sf)

        print("pts_num", pts_num, "seg_num", seg_num)

        if pts_num == seg_num:

            # cut block
            bbox = pc_getbbox(pf)
            split_x = []
            split_y = []
            block_list = []

            dim = [bbox[1] - bbox[0], bbox[3] - bbox[2], bbox[5] - bbox[4]]
            # compute split x
            if dim[0] > max_b:
                block_num = int(dim[0] / (max_b - overlap))
                for c in range(block_num):
                    split_x.append([c * (max_b - overlap), c * (max_b - overlap) + max_b])
            else:
                split_x.append([0, dim[0]])

            # compute split y
            if dim[1] > max_b:
                block_num = int(dim[1] / (max_b - overlap))
                for c in range(block_num):
                    split_y.append([c * (max_b - overlap), c * (max_b - overlap) + max_b])
            else:
                split_y.append([0, dim[1]])

            for px in split_x:
                for py in split_y:
                    block_list.append([px[0] + bbox[0], py[0] + bbox[2], px[1] + bbox[0], py[1] + bbox[2]])

            # split to blocks
            block_indices = {}
            block_len = []
            block_needmerge = []

            for k, block in enumerate(block_list):

                print("Process block", k, block)

                block_indices[k] = []

                for i, p in enumerate(pf):

                    if p[0] >= block[0] and p[0] <= block[2] and p[1] >= block[1] and p[1] <= block[3]:
                        block_indices[k].append(i)

                    temp_list = []
                    if len(block_indices[k]) > max_pts:
                        samplelist = random.sample(range(len(block_indices[k]) - 1), max_pts)
                        for m in samplelist:
                            temp_list.append(block_indices[k][m])
                        block_indices[k] = temp_list
                    else:
                        pass

                block_len.append(len(block_indices[k]))

                if block_len[-1] < block_min_pnum:
                    block_needmerge.append([k, block, block_len[-1]])

            print("reblock")

            # reblock
            for block_nm in block_needmerge:

                print("Merge block:", block_nm)

                if block_nm[2] == 0:
                    block_nm.append(-1)

                else:

                    # compute the nearest block to merge
                    block_i = block_nm[0]
                    dis_list = []
                    x_sum = 0
                    y_sum = 0

                    for n in block_indices[block_i]:
                        x_sum = x_sum + pf[n][0]
                        y_sum = y_sum + pf[n][1]

                    x_avg = x_sum / block_nm[2]
                    y_avg = y_sum / block_nm[2]

                    for block in block_list:
                        block_center = [(block[2] + block[0]) / 2, (block[3] + block[1]) / 2]
                        dis = math.sqrt((block_center[0] - x_avg) ** 2 + (block_center[1] - y_avg) ** 2)
                        dis_list.append(dis)

                    merge_block = dis_list.index(min(dis_list))
                    block_nm.append(merge_block)

            trans_list = []
            out_trs = out_trans + category.split("_", 1)[0] + "/" + category.split(".", 1)[0] + ".trs"

            # save block
            for k, block in enumerate(block_list):

                save_list = [k]

                for block_nm in block_needmerge:

                    if k == block_nm[0]:
                        save_list = [block_nm[3], k]
                    if k == block_nm[3]:
                        save_list = [k, block_nm[0]]

                if save_list[0] == -1:
                    print("zero block")
                    continue

                save_id = category.split(".", 1)[0] + "%05d" % save_list[0]
                out_pts = out_data + category.split("_", 1)[0] + "/" + save_id + ".pts"
                out_seg = out_label + category.split("_", 1)[0] + "/" + save_id + ".seg"

                pf_block = []
                sf_block = []

                for save_k in save_list:

                    for n in block_indices[save_k]:
                        pf_block.append(pf[n])
                        sf_block.append(sf[n])

                bbox_block = pc_getbbox(pf_block)
                trans = [(bbox_block[1] - bbox_block[0]) / 2 + bbox_block[0],
                         (bbox_block[3] - bbox_block[2]) / 2 + bbox_block[2], bbox_block[4]]
                trans_list.append([save_id, trans])

                with open(out_pts, "w") as f:
                    for pt in pf_block:
                        f.writelines(str((pt[0] - trans[0])) + " " + str((pt[2] - trans[2])) + " " + str(
                            (pt[1] - trans[1])) + " " +
                             str(float(pt[3]) / 255 - 0.5) + " " + str(float(pt[4]) / 255 - 0.5) + " " +
                             str(float(pt[5]) / 255 - 0.5) + "\n")
                print("save pts", out_pts, len(pf_block))

                with open(out_seg, "w") as f:
                    for s in sf_block:
                        f.writelines(str(s) + "\n")
                print("save seg", out_seg, len(sf_block))

            # save trans
            with open(out_trs, "w") as f_w:
                for trans in trans_list:
                    f_w.writelines(trans[0])
                    for t in trans[1]:
                        f_w.writelines(" " + str(t))
                    f_w.writelines("\n")
            print("save trans", out_trs)

        else:
            print("pts_num != seg_num!")
            os._exit(0)


unpickle(npy_file_train, train_data_root, train_label_root, train_trans_root)
unpickle(npy_file_val,val_data_root,val_label_root,val_trans_root)