#!/usr/bin/python3
'''Read Pointnet++ Scannet .pickle file'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import os
import random
import math
import numpy as np

pkl_file_test = open('scannet_test.pickle', 'rb')
pkl_file_train = open('scannet_train.pickle', 'rb')

classid = "01"

# block length/overlap/edge
max_l = 1.5
overlap = 0.75
edge = 0.3

# block merge threshold
block_min_pnum = 600

# out path
out_root = './scannet_split_dataset/'

train_data_root = out_root + '/train_data/' + classid + "/"
train_label_root = out_root + '/train_label/' + classid + "/"
train_trans_root = out_root + '/train_trans/' + classid + "/"

test_data_root = out_root + 'val_data/' + classid + '/'
test_label_root = out_root + 'val_label/' + classid + '/'
test_trans_root = out_root + 'val_trans/' + classid + '/'


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


def unpickle(pkl_file, out_data, out_label, out_trans):
    pts = pickle.load(pkl_file)
    seg = pickle.load(pkl_file)
    pkl_file.close()

    pts_file_num = len(pts)
    seg_file_num = len(seg)
    print("pts_file_num", pts_file_num)
    print("seg_file_num", seg_file_num)

    # check the sample num
    if pts_file_num == seg_file_num:

        f_num = pts_file_num

        # check the path
        if not os.path.exists(out_data):
            print(out_data, "Not Exists! Create", out_data)
            os.makedirs(out_data)
        if not os.path.exists(out_label):
            print(out_label, "Not Exists! Create", out_label)
            os.makedirs(out_label)
        if not os.path.exists(out_trans):
            print(out_trans, "Not Exists! Create", out_trans)
            os.makedirs(out_trans)

    else:

        print("pts_file_num != seg_file_num!")
        os._exit(0)

    # iter sample
    for index in range(f_num):

        print("\nProcess", index)

        pf = pts[index]
        sf = seg[index]
        pts_num = len(pf)
        seg_num = len(sf)

        # check the pts and seg num
        if pts_num == seg_num:

            # split blocks
            split_x = []
            split_y = []
            block_list = []

            bbox = pc_getbbox(pf)

            dim = [bbox[1] - bbox[0], bbox[3] - bbox[2], bbox[5] - bbox[4]]
            max_dim = max(dim)

            # compute split x
            if dim[0] > max_l:
                block_num = int(dim[0] / (max_l - overlap))
                for c in range(block_num):
                    split_x.append([c * (max_l - overlap), c * (max_l - overlap) + max_l])
            else:
                split_x.append([0, dim[0]])

            # compute split y
            if dim[1] > max_l:
                block_num = int(dim[1] / (max_l - overlap))
                for c in range(block_num):
                    split_y.append([c * (max_l - overlap), c * (max_l - overlap) + max_l])
            else:
                split_y.append([0, dim[1]])

            # compute block_list
            for px in split_x:
                for py in split_y:
                    block_list.append([px[0] + bbox[0], py[0] + bbox[2], px[1] + bbox[0], py[1] + bbox[2]])

            # split to blocks
            block_indices = {}
            block_edge_indices = {}
            block_len = []
            block_needmerge = []

            for k, block in enumerate(block_list):

                print("Process block", k, block)

                block_indices[k] = []
                block_edge_indices[k] = []

                for i, p in enumerate(pf):

                    if p[0] >= block[0] and p[0] <= block[2] and p[1] >= block[1] and p[1] <= block[3]:
                        block_indices[k].append(i)

                    elif p[0] >= (block[0] - edge) and p[0] <= (block[2] + edge) and p[1] >= (block[1] - edge) and p[
                        1] <= (block[3] + edge):
                        block_edge_indices[k].append(i)

                block_len.append(len(block_indices[k]))

                if block_len[-1] < block_min_pnum:
                    block_needmerge.append([k, block, block_len[-1]])

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

            # save trans file (Camera CS to Block CS)
            trans_list = []
            out_trs = out_trans + "%05d" % index + ".trs"

            # save block
            for k, block in enumerate(block_list):

                save_list = [k]

                for block_nm in block_needmerge:

                    if k == block_nm[0]:
                        save_list = [block_nm[3], k]
                    if k == block_nm[3]:
                        save_list = [k, block_nm[0]]
                # zero block
                if save_list[0] == -1:
                    continue

                save_id = "%05d" % index + "%03d" % save_list[0]
                out_pts = out_data + save_id + ".pts"
                out_seg = out_label + save_id + ".seg"

                pf_block = []
                sf_block = []

                for save_k in save_list:

                    for n in block_indices[save_k]:
                        pf_block.append(pf[n])
                        sf_block.append(sf[n])

                    for n in block_edge_indices[save_k]:
                        pf_block.append(pf[n])
                        sf_block.append(0)

                bbox_block = pc_getbbox(pf_block)
                trans = [(bbox_block[1] - bbox_block[0]) / 2 + bbox_block[0],
                         (bbox_block[3] - bbox_block[2]) / 2 + bbox_block[2], bbox_block[4]]
                trans_list.append([save_id, trans])

                with open(out_pts, "w") as f:
                    for pt in pf_block:
                        f.writelines(str((pt[0] - trans[0])) + " " + str((pt[2] - trans[2])) + " " + str(
                            (pt[1] - trans[1])) + "\n")
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


if __name__ == '__main__':
    # read and split train
    unpickle(pkl_file_train, train_data_root, train_label_root, train_trans_root)
    # read and split test
    unpickle(pkl_file_test, test_data_root, test_label_root, test_trans_root)