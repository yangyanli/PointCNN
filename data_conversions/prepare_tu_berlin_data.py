#!/usr/bin/python3
'''Convert TU-Berlin sketches to points with normals.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import h5py
import math
import random
import argparse
import numpy as np
from datetime import datetime
from numpy import linalg as LA
from scipy.spatial import ConvexHull
from svgpathtools import svg2paths, Path, Line, CubicBezier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils


def moving_least_square_with_rigid_transformation(p, q, v, r):
    w = 1 / (LA.norm(p - v, axis=-1, ord=2, keepdims=True) ** 0.5 + r)
    w_sum = np.sum(w)

    p_star = np.sum(w * p, axis=0, keepdims=True) / w_sum  # (1, 2)
    p_hat = np.expand_dims(p - p_star, axis=1)  # (N, 1, 2)
    p_hat_x = np.concatenate([-p_hat[..., np.newaxis, 1], p_hat[..., np.newaxis, 0]], axis=-1)

    p_p = np.concatenate([p_hat, -p_hat_x], axis=-2)  # (N, 2, 2)
    v_p_star = v - p_star  # (1, 2)
    v_p_star_x = np.concatenate([-v_p_star[..., np.newaxis, 1], v_p_star[..., np.newaxis, 0]], axis=-1)
    vp_vp = np.transpose(np.concatenate([v_p_star, -v_p_star_x], axis=-2))  # (2, 2)
    A = np.expand_dims(w, axis=-1) * np.matmul(p_p, np.expand_dims(vp_vp, axis=0))  # (N, 2, 2)

    q_star = np.sum(w * q, axis=0, keepdims=True) / w_sum  # (1, 2)
    q_hat = np.expand_dims(q - q_star, axis=1)  # (N, 1, 2)

    fr_arrow_v = np.sum(np.matmul(q_hat, A), axis=0)  # (1, 2)
    fr_v = (LA.norm(v_p_star, axis=-1, ord=2) / (LA.norm(fr_arrow_v, axis=-1, ord=2) + 1e-6)) * fr_arrow_v + q_star
    return fr_v[0, 0], fr_v[0, 1]


def augment(path_nested, num):
    path_list = []

    path = Path()
    for p in path_nested:
        for segment in p:
            path.append(segment)

    end_points_list = []
    for segment in path:
        s = segment.bpoints()[0]
        e = segment.bpoints()[-1]
        end_points_list.append((s.real, s.imag))
        end_points_list.append((e.real, e.imag))
    end_points = np.array(end_points_list)
    hull_points = end_points[ConvexHull(end_points).vertices]
    idx_xmin, idx_ymin = np.argmin(hull_points, axis=0)
    idx_xmax, idx_ymax = np.argmax(hull_points, axis=0)
    x_range = 0.15 * (hull_points[idx_xmax][0] - hull_points[idx_xmin][0])
    y_range = 0.15 * (hull_points[idx_ymax][1] - hull_points[idx_ymin][1])
    idx_min_max = np.unique([idx_xmin, idx_ymin, idx_xmax, idx_ymax])

    for _ in range(num):
        # global deformation
        p = hull_points
        q = hull_points.copy()
        for idx in idx_min_max:
            x, y = p[idx]
            q[idx] = (x + random.gauss(0, x_range), y + y_range * random.gauss(0, y_range))

        path_deformed = Path()
        for segment in path:
            points = []
            for v in segment.bpoints():
                real, imag = moving_least_square_with_rigid_transformation(p, q, np.array([v.real, v.imag]),
                                                                           max(x_range, y_range))
                point_xformed = complex(real, imag)
                points.append(point_xformed)
            if len(segment.bpoints()) == 2:
                line = Line(points[0], points[1])
                path_deformed.append(line)
            else:
                cubic_bezier = CubicBezier(points[0], points[1], points[2], points[3])
                path_deformed.append(cubic_bezier)

        path_list.append(path_deformed)

    return path_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', help='Path to data folder')
    parser.add_argument('--point_num', '-p', help='Point number for each sample', type=int, default=1024)
    parser.add_argument('--save_ply', '-s', help='Convert .pts to .ply', action='store_true')
    parser.add_argument('--augment', '-a', help='Data augmentation', action='store_true')
    parser.add_argument('--create-train-test',
                        help='Concatenate file lists to generate train_files.txt and test_files.txt',
                        action='store_true')
    args = parser.parse_args()
    print(args)

    batch_size = 2048
    fold_num = 3

    tag_aug = '_ag' if args.augment else ''

    folder_svg = args.folder if args.folder else '../../data/tu_berlin/svg'
    root_folder = os.path.dirname(folder_svg)
    folder_pts = os.path.join(root_folder, 'pts' + tag_aug)
    filelist_svg = [line.strip() for line in open(os.path.join(folder_svg, 'filelist.txt'))]

    category_label = dict()
    with open(os.path.join(os.path.dirname(folder_svg), 'categories.txt'), 'w') as file_categories:
        for filename in filelist_svg:
            category = os.path.split(filename)[0]
            if category not in category_label:
                file_categories.write('%s %d\n' % (category, len(category_label)))
                category_label[category] = len(category_label)

    filelist_svg_failed = []
    data = np.zeros((batch_size, args.point_num, 6))
    label = np.zeros((batch_size), dtype=np.int32)
    for idx_fold in range(fold_num):
        filelist_svg_fold = [filename for i, filename in enumerate(filelist_svg) if i % fold_num == idx_fold]
        random.seed(idx_fold)
        random.shuffle(filelist_svg_fold)

        filename_filelist_svg_fold = os.path.join(root_folder, 'filelist_fold_%d.txt' % (idx_fold))
        if os.path.exists(filename_filelist_svg_fold):
            print('{}-{} exists, skipping'.format(datetime.now(), filename_filelist_svg_fold))
            continue
        with open(filename_filelist_svg_fold, 'w') as filelist_svg_fold_file:
            for filename in filelist_svg_fold:
                filelist_svg_fold_file.write('%s\n' % (filename))

        idx_h5 = 0
        idx = 0
        filename_filelist_h5 = os.path.join(root_folder, 'fold_%d_files%s.txt' % (idx_fold, tag_aug))
        with open(filename_filelist_h5, 'w') as filelist_h5_file:
            for idx_file, filename in enumerate(filelist_svg_fold):
                filename_svg = os.path.join(folder_svg, filename)
                try:
                    paths, attributes = svg2paths(filename_svg)
                except:
                    filelist_svg_failed.append(filename_svg)
                    print('{}-Failed to parse {}!'.format(datetime.now(), filename_svg))
                    continue

                points_array = np.zeros(shape=(args.point_num, 3), dtype=np.float32)
                normals_array = np.zeros(shape=(args.point_num, 3), dtype=np.float32)

                path = Path()
                for p in paths:
                    p_non_empty = Path()
                    for segment in p:
                        if segment.length() > 0:
                            p_non_empty.append(segment)
                    if len(p_non_empty) != 0:
                        path.append(p_non_empty)

                path_list = []
                if args.augment:
                    for removal_idx in range(6):
                        path_with_removal = Path()
                        for p in path[:math.ceil((0.4 + removal_idx * 0.1) * len(paths))]:
                            path_with_removal.append(p)
                        path_list.append(path_with_removal)
                    path_list = path_list + augment(path, 6)
                else:
                    path_list.append(path)

                for path_idx, path in enumerate(path_list):
                    for sample_idx in range(args.point_num):
                        sample_idx_float = (sample_idx + random.random()) / (args.point_num - 1)
                        while True:
                            try:
                                point = path.point(sample_idx_float)
                                normal = path.normal(sample_idx_float)
                                break
                            except:
                                sample_idx_float = random.random()
                                continue
                        points_array[sample_idx] = (point.real, sample_idx_float, point.imag)
                        normals_array[sample_idx] = (normal.real, random.random() * 1e-6, normal.imag)

                    points_min = np.amin(points_array, axis=0)
                    points_max = np.amax(points_array, axis=0)
                    points_center = (points_min + points_max) / 2
                    scale = np.amax(points_max - points_min) / 2
                    points_array = (points_array - points_center) * (0.8 / scale, 0.4, 0.8 / scale)

                    if args.save_ply:
                        tag_aug_idx = tag_aug + '_' + str(path_idx) if args.augment else tag_aug
                        filename_pts = os.path.join(folder_pts, filename[:-4] + tag_aug_idx + '.ply')
                        data_utils.save_ply(points_array, filename_pts, normals=normals_array)

                    idx_in_batch = idx % batch_size
                    data[idx_in_batch, ...] = np.concatenate((points_array, normals_array), axis=-1).astype(np.float32)
                    label[idx_in_batch] = category_label[os.path.split(filename)[0]]
                    if ((idx + 1) % batch_size == 0) \
                            or (idx_file == len(filelist_svg_fold) - 1 and path_idx == len(path_list) - 1):
                        item_num = idx_in_batch + 1
                        filename_h5 = 'fold_%d_%d%s.h5' % (idx_fold, idx_h5, tag_aug)
                        print('{}-Saving {}...'.format(datetime.now(), os.path.join(root_folder, filename_h5)))
                        filelist_h5_file.write('./%s\n' % (filename_h5))

                        file = h5py.File(os.path.join(root_folder, filename_h5), 'w')
                        file.create_dataset('data', data=data[0:item_num, ...])
                        file.create_dataset('label', data=label[0:item_num, ...])
                        file.close()

                        idx_h5 = idx_h5 + 1
                    idx = idx + 1

    if len(filelist_svg_failed) != 0:
        print('{}-Failed to parse {} sketches!'.format(datetime.now(), len(filelist_svg_failed)))

    if args.create_train_test:
        print('{}-Generating train_files.txt and test_files.txt'.format(datetime.now()))
        train_files = open(os.path.join(root_folder, "train_files.txt"), "w")
        test_files = open(os.path.join(root_folder, "test_files.txt"), "w")
        with train_files, test_files:
            for idx_fold in range(fold_num):
                filename = os.path.join(root_folder, 'fold_%d_files%s.txt' % (idx_fold, tag_aug))
                contents = open(filename, "r").read()
                # Use folders 0..N-1 for train and N for test
                if idx_fold < fold_num - 1:
                    train_files.write(contents)
                else:
                    test_files.write(contents)

if __name__ == '__main__':
    main()
