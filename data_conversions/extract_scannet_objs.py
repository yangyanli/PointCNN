#!/usr/bin/python3
"""Convert original Scannet data to PointCNN Classification dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import plyfile
from plyfile import PlyData
import argparse
import numpy as np


def dir(root, type='f', addroot=True):
    dirList = []
    fileList = []
    root = root + "/"
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

    print("save ply to", filename)


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


def scene2instances(scene_path, out_root, all_label, label_map, issave_ply):
    print("Process Scene:", scene_path)

    sceneid = scene_path.strip().split("scene")[1]
    spaceid = sceneid.split("_")[0]
    scanid = sceneid.split("_")[1]

    label_list = all_label[0]
    label_info = all_label[1]

    pts_dir = out_root + "/pts/"

    # check the path
    if not os.path.exists(pts_dir):
        print(pts_dir, "Not Exists! Create", pts_dir)
        os.makedirs(pts_dir)

    if save_ply:

        ply_dir = out_root + "/ply/" + spaceid + scanid + "/"

        if not os.path.exists(ply_dir):
            print(ply_dir, "Not Exists! Create", ply_dir)
            os.makedirs(ply_dir)

    ply_file = scene_path + "/scene" + sceneid + "_vh_clean_2.ply"
    jsonflie = scene_path + "/scene" + sceneid + "_vh_clean_2.0.010000.segs.json"
    aggjsonfile = scene_path + "/scene" + sceneid + ".aggregation.json"

    # Read ply file
    print("\nRead ply file:", ply_file)
    plydata = PlyData.read(ply_file).elements[0].data
    pts_num = len(plydata)
    print("points num:", pts_num)

    # Read json file
    print("Read json file:", jsonflie)
    json_data = json.load(open(jsonflie))

    # check json file
    if json_data['sceneId'].strip() == ('scene' + sceneid):

        segIndices = json_data['segIndices']
        seg_num = len(segIndices)

        # check num
        if seg_num != pts_num:
            print("seg num != pts num!")
            os.exit(0)

    else:

        print("Read Wrong Json File!")
        os.exit(0)

    # Read aggregation json file
    print("Read aggregation json file:", aggjsonfile)
    aggjson_data = json.load(open(aggjsonfile))

    # check json file
    if aggjson_data['sceneId'].strip() == ('scene' + sceneid):

        segGroups = aggjson_data['segGroups']

    else:

        print("Read Wrong Aggregation Json File!")
        os.exit(0)

    # split pts
    obj_dict = {}

    for k, pts in enumerate(plydata):

        seg_indice = segIndices[k]

        # find obj
        label = "unannotated"
        objid = -1

        for seg in segGroups:

            segments = seg['segments']

            if seg_indice in segments:
                label = seg['label'].strip()
                objid = seg['objectId']

                break

        obj_key = str(label + "_" + str(objid)).strip()

        if obj_key not in obj_dict.keys():
            obj_dict[obj_key] = []

        obj_dict[obj_key].append(pts)

    # save data file by obj
    for objkey in obj_dict.keys():

        obj_label = objkey.split("_")[0]
        obj_id = objkey.split("_")[1]

        if obj_label in label_list:

            label_id = label_list.index(obj_label)

        else:

            label_id = 0

        label_full = label_info[label_id]

        label_id = label_full[0]
        label_s55 = label_full[3]
        label_s55_id = 0

        for l in label_map:

            if label_s55 == l[1]:
                label_s55_id = l[0]

        if label_s55_id != 0:

            pts_out_file = pts_dir + spaceid + scanid + "%04d" % int(obj_id) + "_" + str(label_s55_id) + ".pts"
            f_pts = open(pts_out_file, "w")

            pts = []
            rgb = []

            for p in obj_dict[objkey]:
                pts.append([p[0], p[1], p[2]])
                rgb.append([p[3], p[4], p[5]])

            bbox = pc_getbbox(pts)
            dimxy = [bbox[1] - bbox[0], bbox[3] - bbox[2]]

            # retrans
            for i in range(len(pts)):
                pts[i] = [pts[i][0] - bbox[0] - dimxy[0] / 2, pts[i][2] - bbox[4], pts[i][1] - bbox[2] - dimxy[1] / 2]

            if issave_ply:

                ply_out_file = ply_dir + spaceid + "_" + scanid + "_" + "%04d" % int(
                    obj_id) + "_" + obj_label + "_" + str(label_id) + "_" + str(label_s55) + "_" + str(
                    label_s55_id) + ".ply"

                seg_pts = []
                seg_ply = []

                for k, p in enumerate(pts):
                    seg_pts.append((p[0], p[1], p[2]))
                    seg_ply.append((rgb[k][0], rgb[k][1], rgb[k][2]))

                save_ply(seg_pts, seg_ply, ply_out_file)

            # write pts xyzrgb
            for k, p in enumerate(pts):
                f_pts.writelines(str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + " " + str(rgb[k][0]) + " " + str(
                    rgb[k][1]) + " " + str(rgb[k][2]) + "\n")

            f_pts.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', help='Path to data folder')
    parser.add_argument('--benchmark', '-b', help='Path to benchmark folder')
    parser.add_argument('--outpath', '-o', help='Path to output folder')
    parser.add_argument('--saveply', '-s', action='store_true', help='Save color ply or not')
    args = parser.parse_args()
    print(args)

    label_tsv = args.benchmark + "/scannet-labels.combined.tsv"
    trainval_list_file = args.benchmark + "/scannet_trainval.txt"
    test_list_file = args.benchmark + "/scannet_test.txt"
    label_shapenetcore55 = args.benchmark + "/classes_ObjClassification-ShapeNetCore55.txt"

    ##########################################################Read Source##########################################################

    print("read scene dir:", args.folder)
    scenedir = dir(args.folder, 'd')

    print("read trainval list:", trainval_list_file)
    train_scene_list = []
    with open(trainval_list_file, 'r') as train_f:
        for line in train_f.readlines():
            sceneid = line.strip().split("scene")[1]
            spaceid = sceneid.split("_")[0]
            scanid = sceneid.split("_")[1]
            train_scene_list.append(spaceid + scanid)

    print("read test list:", test_list_file)
    test_scene_list = []
    with open(test_list_file, 'r') as train_f:
        for line in train_f.readlines():
            sceneid = line.strip().split("scene")[1]
            spaceid = sceneid.split("_")[0]
            scanid = sceneid.split("_")[1]
            test_scene_list.append(spaceid + scanid)

    print("read label tsv file:", label_tsv)
    label_map = []
    label_info = []

    with open(label_tsv, 'r') as tsv_f:

        for k, line in enumerate(tsv_f.readlines()):

            if k > 0:
                line_s = line.strip().split('\t')

                label_id = int(line_s[0])
                category = line_s[1]
                count = int(line_s[2])
                ShapeNetCore55 = line_s[11]

                label_map.append(category)
                label_info.append([label_id, category, count, ShapeNetCore55])

    print("read shapenetcore55 label file:", label_shapenetcore55)
    label_shapenetcore55_map = []
    with open(label_shapenetcore55, 'r') as label_shapenetcore55_f:

        for k, line in enumerate(label_shapenetcore55_f.readlines()):
            line_s = line.strip().split('\t')
            label_id = line_s[0]
            category = line_s[1]
            label_shapenetcore55_map.append([label_id, category])

    # split scene to train and test
    process_train_list = []
    process_test_list = []

    for scene in scenedir:

        sceneid = scene.strip().split("scene")[1]
        spaceid = sceneid.split("_")[0]
        scanid = sceneid.split("_")[1]
        scenename = spaceid + scanid

        if scenename in train_scene_list:

            process_train_list.append(scene)

        elif scenename in test_scene_list:

            process_test_list.append(scene)

    print("Train all:", len(train_scene_list), "Test all:", len(test_scene_list), "Dir all:", len(scenedir))
    print("Process Train:", len(process_train_list), "Process Test:", len(process_test_list))

    ##########################################################Process Data##########################################################
    print("Process Train Scene:")

    for scene in process_train_list:
        scene2instances(scene, args.outpath + "/train/", [label_map, label_info], label_shapenetcore55_map,
                        args.saveply)

    print("Process Test Scene:")

    for scene in process_test_list:
        scene2instances(scene, args.outpath + "/test/", [label_map, label_info], label_shapenetcore55_map, args.saveply)


if __name__ == '__main__':
    main()
