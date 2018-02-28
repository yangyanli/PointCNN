#!/usr/bin/python3

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
        return fileList

    elif type == "d":
        return dirList

    else:
        print("ERROR: TMC.dir(root,type) type must be [f] for file or [d] for dir")

        return 0


def save_ply(points, colors, filename):
    vertex = np.array([tuple(p) for p in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    vertex_color = np.array([tuple(c) for c in colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    n = len(vertex)
    assert len(vertex_color) == n

    vertex_all = np.empty(n, dtype=vertex.dtype.descr + vertex_color.dtype.descr)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    for prop in vertex_color.dtype.names:
        vertex_all[prop] = vertex_color[prop]

    ply = plyfile.PlyData([plyfile.PlyElement.describe(vertex_all, 'vertex')], text=False)
    ply.write(filename)


def voxel_downsample(pts, seg, res):
    coordmax = np.max(pts, axis=0)
    coordmin = np.min(pts, axis=0)

    nvox = np.ceil((coordmax - coordmin) / res)
    vidx = np.ceil((pts - coordmin) / res)
    vidx = vidx[:, 0] + vidx[:, 1] * nvox[0] + vidx[:, 2] * nvox[0] * nvox[1]

    uvidx, vpidx = np.unique(vidx, return_index=True)
    # compute voxel label
    return np.array(pts)[vpidx], np.array(seg)[vpidx]


def voxel_upsample(seg, ori_pts, res):
    ori_seg = []

    coordmax = np.max(ori_pts, axis=0)
    coordmin = np.min(ori_pts, axis=0)

    nvox = np.ceil((coordmax - coordmin) / res)
    vidx = np.ceil((ori_pts - coordmin) / res)
    vidx = vidx[:, 0] + vidx[:, 1] * nvox[0] + vidx[:, 2] * nvox[0] * nvox[1]

    uvidx, vpidx = np.unique(vidx, return_index=True)

    print(len(uvidx), len(seg))

    voxel_label_dict = {}

    for k, vid in enumerate(uvidx):
        voxel_label_dict[vid] = seg[k]

    for vid in vidx:
        ori_seg.append(voxel_label_dict[vid])

    return ori_pts, ori_seg


def EuclideanDistances(A, B):
    A = np.array(A)
    B = np.array(B)

    BT = B.transpose()

    vecProd = np.dot(A, BT)
    SqA = A ** 2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))

    SqB = B ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    return ED


def nearest_upsample(pts, seg, ori_pts):
    dis = EuclideanDistances(ori_pts, pts)


def seg2color(seg):
    color_list = [(192, 153, 110), (188, 199, 253), (214, 255, 0), (159, 0, 142), (153, 255, 85), (118, 79, 2),
                  (123, 72, 131), (2, 176, 127), (1, 126, 184), (0, 144, 161), (106, 107, 128), (254, 230, 0),
                  (0, 255, 255), (255, 167, 254), (233, 93, 189), (0, 100, 0), (132, 169, 1), (150, 0, 61),
                  (188, 136, 0), (0, 0, 255)]
    abn_color = (0, 0, 0)
    color = []

    for s in seg:
        color.append(color_list[s])
    return color

###########need to modify paths when you run this code#############
pts_file_root = "../../../data/S3DIS//out_part_rgb/test/train_data/Area6_data/01"
seg_file_root = "../../../data/S3DIS//pred/Area6_data/"
out_seg_root = "../../../data/S3DIS//upsampling/upsample_pred_A6/seg/"
out_ply_root = "../../../data/S3DIS//upsample_pred_A6/ply/"

if not os.path.exists(out_seg_root):
    print(out_seg_root, "Not Exists! Create", out_seg_root)
    os.makedirs(out_seg_root)

if not os.path.exists(out_ply_root):
    print(out_ply_root, "Not Exists! Create", out_ply_root)
    os.makedirs(out_ply_root)

seg_file_list = dir(seg_file_root, 'f', False)

seg_files = []
pts_files = []
out_segs = []
out_plys = []

for pred_seg in seg_file_list:
    seg_files.append(seg_file_root + "/" + pred_seg)
    pts_files.append(pts_file_root + "/" + pred_seg.replace(".seg", ".pts"))

    out_segs.append(out_seg_root + "/" + pred_seg)
    # out_plys.append(out_ply_root + "/" + pred_seg.replace(".seg",".ply"))

# out_ply = "./show.ply"
# out_ply_dsample = "./show_dsample.ply"
# out_ply_usample = "./show_usample.ply"

for k, seg_f in enumerate(seg_files):

    print("Process:", k, "/", len(seg_files))

    seg = []
    ori_pts = []

    # read pts
    print("Read pts:", pts_files[k])
    with open(pts_files[k], 'r') as pts_f:

        for line in pts_f:
            line_s = line.strip().split(" ")

            ori_pts.append((float(line_s[0]), float(line_s[1]), float(line_s[2])))

    # read seg
    print("Read seg:", seg_files[k])
    with open(seg_files[k], 'r') as seg_f:

        for line in seg_f:
            line_s = int(line.strip())

            seg.append(line_s)

    print("Up sample...")
    pts_upsample, seg_upsample = voxel_upsample(seg, ori_pts, 0.05)

    print("Save upsampled seg:", out_segs[k])
    with open(out_segs[k], 'w') as seg_wf:

        for s in seg_upsample:
            seg_wf.writelines(str(s) + "\n")

    # print "Save ply:",out_plys[k]
    # save_ply(pts_upsample,seg2color(seg_upsample),out_plys[k])

