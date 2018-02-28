#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import random
import math
import numpy as np

BASE_DIR = os.path.join(os.path.dirname(__file__), 'prepare_label_rgb')

res = 0.05  # 5cm
max_b = 1.5
overlap = 0.75
edge = 0.3
part_min_pnum = 600
sample_flag = False

# out path
train_data_root = "./out_part/train_data_downsampling/"
train_label_root = "./out_part/train_label_downsampling/"
train_trans_root = "./out_part/train_trans_downsampling/"

test_data_root = "./out_part_rgb/test_data_downsampling/"
test_label_root = "./out_part_rgb/test_data_downsampling/"
test_trans_root = "./out_part_rgb/test_data_downsampling/"


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
    path_Areas = os.listdir(npy_file)
    for Area in path_Areas:
        # check the path
        if not os.path.exists(out_data + "Area" + Area[-1] + "_data/01"):
            print(out_data, "Not Exists! Create", out_data)
            os.makedirs(out_data + "Area" + Area[-1] + "_data/01")
        if not os.path.exists(out_label + "Area" + Area[-1] + "_label/01"):
            print(out_label, "Not Exists! Create", out_label)
            os.makedirs(out_label + "Area" + Area[-1] + "_label/01")
        if not os.path.exists(os.path.join(out_trans, Area)):
            print(out_trans, "Not Exists! Create", out_trans)
            os.makedirs(os.path.join(out_trans, Area))

        path_Rooms = os.listdir(os.path.join(npy_file, Area))
        for Room in path_Rooms:
            path_data = os.path.join(npy_file, Area, Room, "xyzrgb.npy")
            path_seg = os.path.join(npy_file, Area, Room, "label.npy")

            print("\nProcess", path_data)
            #test = np.load(path_data)

            pf = np.load(path_data)
            sf = np.load(path_seg).astype(int)
            sf = sf.reshape(int(sf.shape[0]))

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

                # cut part
                bbox = pc_getbbox(pf)
                part_x = []
                part_y = []
                part_list = []

                dim = [bbox[1] - bbox[0], bbox[3] - bbox[2], bbox[5] - bbox[4]]
                max_dim = max(dim)

                print("bbox:", bbox)
                print("PC Dim:", dim)

                # part x
                if dim[0] > max_b:

                    part_num = int(dim[0] / (max_b - overlap))

                    for c in range(part_num):
                        part_x.append([c * (max_b - overlap), c * (max_b - overlap) + max_b])

                else:

                    part_x.append([0, dim[0]])

                # part y
                if dim[1] > max_b:

                    part_num = int(dim[1] / (max_b - overlap))

                    for c in range(part_num):
                        part_y.append([c * (max_b - overlap), c * (max_b - overlap) + max_b])

                else:

                    part_y.append([0, dim[1]])

                for px in part_x:

                    for py in part_y:
                        part_list.append([px[0] + bbox[0], py[0] + bbox[2], px[1] + bbox[0], py[1] + bbox[2]])

                print("part_x", part_x)
                print("part_y", part_y)

                part_indices = {}
                part_edge_indices = {}
                part_len = []
                part_needmerge = []

                for k, part in enumerate(part_list):

                    print("Process part", k, part)

                    part_indices[k] = []
                    part_edge_indices[k] = []

                    for i, p in enumerate(pf):

                        if p[0] >= part[0] and p[0] <= part[2] and p[1] >= part[1] and p[1] <= part[3]:

                            part_indices[k].append(i)

                        elif p[0] >= (part[0] - edge) and p[0] <= (part[2] + edge) and p[1] >= (part[1] - edge) and p[
                            1] <= (part[3] + edge):

                            part_edge_indices[k].append(i)

                    part_len.append(len(part_indices[k]))

                    print("center", part_len[-1], "edge", len(part_edge_indices[k]))

                    if part_len[-1] < part_min_pnum:
                        part_needmerge.append([k, part, part_len[-1]])

                print("repart")

                # repart
                for part_nm in part_needmerge:

                    print(part_nm)

                    if part_nm[2] == 0:

                        part_nm.append(-1)

                    else:

                        part_i = part_nm[0]

                        dis_list = []
                        x_sum = 0
                        y_sum = 0

                        for n in part_indices[part_i]:
                            x_sum = x_sum + pf[n][0]
                            y_sum = y_sum + pf[n][1]

                        x_avg = x_sum / part_nm[2]
                        y_avg = y_sum / part_nm[2]

                        for part in part_list:
                            part_center = [(part[2] + part[0]) / 2, (part[3] + part[1]) / 2]

                            dis = math.sqrt((part_center[0] - x_avg) ** 2 + (part_center[1] - y_avg) ** 2)

                            dis_list.append(dis)

                        merge_part = dis_list.index(min(dis_list))

                        print("merge_part", merge_part)

                        part_nm.append(merge_part)

                trans_list = []
                out_trs = out_trans + Area + "/" + Room + ".trs"

                # save part
                for k, part in enumerate(part_list):

                    print("save part", k)

                    save_list = [k]

                    for part_nm in part_needmerge:

                        if k == part_nm[0]:
                            save_list = [part_nm[3], k]

                        if k == part_nm[3]:
                            save_list = [k, part_nm[0]]

                    print("save_list", save_list)

                    if save_list[0] == -1:
                        print("zero part")
                        continue

                    save_id = Room + "%03d" % save_list[0]

                    out_pts = out_data + "Area" + Area[-1] + "_data/01/" + save_id + ".pts"
                    out_seg = out_label + "Area" + Area[-1] + "_label/01/" + save_id + ".seg"

                    pf_part = []
                    sf_part = []

                    for save_k in save_list:

                        for n in part_indices[save_k]:
                            pf_part.append(pf[n])
                            sf_part.append(sf[n])

                    bbox_part = pc_getbbox(pf_part)
                    trans = [(bbox_part[1] - bbox_part[0]) / 2 + bbox_part[0],
                             (bbox_part[3] - bbox_part[2]) / 2 + bbox_part[2], bbox_part[4]]
                    print("save pts", out_pts, len(pf_part))
                    print("trans", trans)
                    trans_list.append([save_id, trans])

                    with open(out_pts, "w") as f:

                        for pt in pf_part:
                            # f.writelines(str((pt[0]-trans[0])) + " " + str((pt[2]-trans[2])) + " " + str((pt[1]-trans[1])) + "\n")
                            f.writelines(str((pt[0] - trans[0])) + " " + str((pt[2] - trans[2])) + " " + str(
                                (pt[1] - trans[1])) + " " +
                                         str(float(pt[3]) / 255 - 0.5) + " " + str(
                                float(pt[4]) / 255 - 0.5) + " " + str(float(pt[5]) / 255 - 0.5) + "\n")
                    print("save seg", out_seg, len(sf_part))

                    with open(out_seg, "w") as f:
                        for s in sf_part:
                            f.writelines(str(s) + "\n")

                # save trans
                with open(out_trs, "w") as f_w:
                    for trans in trans_list:
                        f_w.writelines(trans[0])
                        for t in trans[1]:
                            f_w.writelines(" " + str(t))
                        f_w.writelines("\n")

            else:

                print("pts_num != seg_num!")
                os._exit(0)

if __name__ == '__main__':
    # read and split train
    unpickle(BASE_DIR, train_data_root, train_label_root, train_trans_root)
    # read and split test
    unpickle(BASE_DIR, test_data_root, test_label_root, test_trans_root)