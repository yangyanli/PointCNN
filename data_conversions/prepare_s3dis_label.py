#!/usr/bin/python()
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np

DEFAULT_DATA_DIR = '../../data/Stanford3dDataset_v1.2_Aligned_Version'
DEFAULT_OUTPUT_DIR = '../../data/S3DIS/prepare_label_rgb'

p = argparse.ArgumentParser()
p.add_argument(
    "-d", "--data", dest='data_dir',
    default=DEFAULT_DATA_DIR,
    help="Path to S3DIS data (default is %s)" % DEFAULT_DATA_DIR)
p.add_argument(
    "-f", "--folder", dest='output_dir',
    default=DEFAULT_OUTPUT_DIR,
    help="Folder to write labels (default is %s)" % DEFAULT_OUTPUT_DIR)

args = p.parse_args()

object_dict = {
            'clutter':   0,
            'ceiling':   1,
            'floor':     2,
            'wall':      3,
            'beam':      4,
            'column':    5,
            'door':      6,
            'window':    7,
            'table':     8,
            'chair':     9,
            'sofa':     10,
            'bookcase': 11,
            'board':    12}

path_dir_areas =  os.listdir(args.data_dir)

for area in path_dir_areas:
    path_area = os.path.join(args.data_dir, area)
    if not os.path.isdir(path_area):
        continue
    path_dir_rooms = os.listdir(path_area)
    for room in path_dir_rooms:
        path_annotations = os.path.join(args.data_dir, area, room, "Annotations")
        if not os.path.isdir(path_annotations):
            continue
        print(path_annotations)
        path_prepare_label = os.path.join(args.output_dir, area, room)
        if os.path.exists(os.path.join(path_prepare_label, ".labels")):
            print("%s already processed, skipping" % path_prepare_label)
            continue
        xyz_room = np.zeros((1,6))
        label_room = np.zeros((1,1))
        # make store directories
        if not os.path.exists(path_prepare_label):
            os.makedirs(path_prepare_label)
        #############################
        path_objects = os.listdir(path_annotations)
        for obj in path_objects:
            object_key = obj.split("_", 1)[0]
            try:
                val = object_dict[object_key]
            except KeyError:
                continue
            print("%s/%s" % (room, obj[:-4]))
            xyz_object_path = os.path.join(path_annotations, obj)
            try:
                xyz_object = np.loadtxt(xyz_object_path)[:,:] # (N,6)
            except ValueError as e:
                print("ERROR: cannot load %s: %s" % (xyz_object_path, e))
                continue
            label_object = np.tile(val, (xyz_object.shape[0], 1)) # (N,1)
            xyz_room = np.vstack((xyz_room, xyz_object))
            label_room = np.vstack((label_room, label_object))

        xyz_room = np.delete(xyz_room, [0], 0)
        label_room = np.delete(label_room, [0], 0)

        np.save(path_prepare_label+"/xyzrgb.npy", xyz_room)
        np.save(path_prepare_label+"/label.npy", label_room)

        # Marker indicating we've processed this room
        open(os.path.join(path_prepare_label, ".labels"), "w").close()
