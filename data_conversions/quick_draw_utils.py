import os
import sys
import math
import random
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils


def _stoke_decoding(stoke):
    lift_pen_padding = 2.0
    lines = []
    points = []
    x_prev = 0
    y_prev = 0
    was_drawing = False
    for i in range(len(stoke)):
        x = x_prev + stoke[i, 0]
        y = y_prev + stoke[i, 1]
        lift_pen = stoke[i, 2]
        if lift_pen == lift_pen_padding:
            break

        is_drawing = (lift_pen == 0.0)
        if is_drawing:
            points.append((x, y))
        if was_drawing and is_drawing and x_prev != x and y_prev != y:
            lines.append(((x_prev, y_prev), (x, y)))

        x_prev = x
        y_prev = y
        was_drawing = is_drawing
    return lines, points


def map_fn(stoke, label, point_num=512):
    lines, points = _stoke_decoding(stoke)

    points_array = np.zeros(shape=(point_num, 3), dtype=np.float32)
    normals_array = np.zeros(shape=(point_num, 3), dtype=np.float32)
    if len(lines) == 0 and len(points) == 0:
        print('Empty stoke detected!')
    elif len(lines) == 0:
        print('Stoke without any line detected!')
        for sample_idx in range(point_num):
            sample_idx_float = sample_idx / (point_num - 1)
            px, py = points[sample_idx % len(points)]
            points_array[sample_idx] = (px, sample_idx_float, py)
    else:
        line_len_list = []
        for ((x0, y0), (x1, y1)) in lines:
            x_diff = x1 - x0
            y_diff = y1 - y0
            line_len_list.append(math.sqrt(x_diff * x_diff + y_diff * y_diff))
        line_len_sum = sum(line_len_list)
        factor = point_num / line_len_sum
        sample_nums = [math.ceil(line_len * factor) for line_len in line_len_list]
        sample_num_total = sum(sample_nums)
        sample_nums_indices = [x for x, y in sorted(enumerate(sample_nums), key=lambda x: x[1])]
        for i in range(sample_num_total - point_num):
            ii = sample_nums_indices[i]
            sample_nums[ii] = sample_nums[ii] - 1
        assert (sum(sample_nums) == point_num)

        sample_idx = 0
        for idx_line, line_sample_num in enumerate(sample_nums):
            if line_sample_num == 0:
                continue

            ((x0, y0), (x1, y1)) = lines[idx_line]
            nx = y1 - y0
            ny = x0 - x1
            n_len = math.sqrt(nx * nx + ny * ny)
            nx /= n_len
            ny /= n_len
            if line_sample_num == 1:
                sample_idx_float = sample_idx / (point_num - 1)
                points_array[sample_idx] = ((x0 + x1) / 2, sample_idx_float, (y0 + y1) / 2)
                normals_array[sample_idx] = (nx, random.random() * 1e-6, ny)
                sample_idx += 1
            elif line_sample_num > 1:
                x_diff = x1 - x0
                y_diff = y1 - y0
                for alpha in np.linspace(0, 1, line_sample_num):
                    sample_idx_float = sample_idx / (point_num - 1)
                    points_array[sample_idx] = (x0 + alpha * x_diff, sample_idx_float, y0 + alpha * y_diff)
                    normals_array[sample_idx] = (nx, random.random() * 1e-6, ny)
                    sample_idx += 1

    points_min = np.amin(points_array, axis=0)
    points_max = np.amax(points_array, axis=0)
    points_center = (points_min + points_max) / 2
    scale = np.amax(points_max - points_min) / 2
    points_array = (points_array - points_center) * (0.8 / scale, 0.4, 0.8 / scale)

    return np.concatenate((points_array, normals_array), axis=-1).astype(np.float32), label


def _extract_padded_stokes(stokes, stoke_len_max, stoke_placeholder, ratio):
    padded_stokes_list = []
    for stoke in stokes:
        if (len(stoke)) == 0:  # bad data, ignore it!
            continue

        lines, points = _stoke_decoding(stoke)
        if len(lines) == 0 or len(points) == 0:   # bad data, ignore it!
            continue

        pad_len = stoke_len_max - len(stoke)
        if pad_len == 0:
            padded_stokes_list.append(stoke.astype(np.float32))
        else:
            padded_stokes_list.append(np.concatenate([stoke.astype(np.float32), stoke_placeholder[:pad_len]], axis=0))
        if len(padded_stokes_list) > ratio * len(stokes):  # The data is too big, only use a subset...
            break
    return np.stack(padded_stokes_list)


def load_fn(folder_npz, ratio, categories=None):
    lift_pen_padding = 2.0

    categories = [line.strip() for line in
                  open(os.path.join(folder_npz, 'categories.txt'), 'r')] if categories is None else categories

    stoke_len_max = 0
    stoke_len_sum = 0
    stoke_num = 0
    load_data_list = []
    for idx_category, category in enumerate(categories):
        print('{}-Loading category {} ({} of {})...'.format(datetime.now(), category, idx_category+1, len(categories)))
        sys.stdout.flush()
        filename_category = os.path.join(folder_npz, category + '.npz')
        load_data = np.load(filename_category, encoding='bytes')
        load_data_list.append(load_data)
        for tag in load_data:
            for stoke in load_data[tag]:
                stoke_len_max = max(stoke_len_max, stoke.shape[0])
                stoke_len_sum += stoke.shape[0]
            stoke_num += len(load_data[tag])
    print('{}-Max stoke length: {}, average stoke length: {}.'.format(datetime.now(), stoke_len_max,
                                                                      stoke_len_sum / stoke_num))
    sys.stdout.flush()

    stoke_placeholder = np.array([(0.0, 0.0, lift_pen_padding)] * stoke_len_max).astype(np.float32)
    raw_train_list = []
    label_train_list = []
    raw_val_list = []
    label_val_list = []
    for idx_category, category in enumerate(categories):
        print('{}-Extracting category {} ({} of {})...'.format(datetime.now(), category, idx_category+1, len(categories)))
        sys.stdout.flush()

        load_data = load_data_list[idx_category]

        raw_train_list.append(_extract_padded_stokes(load_data['train'], stoke_len_max, stoke_placeholder, ratio))
        label_train_list += [idx_category] * len(raw_train_list[-1])

        raw_val_list.append(_extract_padded_stokes(load_data['valid'], stoke_len_max, stoke_placeholder, ratio))
        label_val_list += [idx_category] * len(raw_val_list[-1])
    raw_train = np.concatenate(raw_train_list, axis=0)
    label_train = np.array(label_train_list)
    raw_val = np.concatenate(raw_val_list, axis=0)
    label_val = np.array(label_val_list)

    print('{}-Shuffling data...'.format(datetime.now()))
    sys.stdout.flush()
    raw_train, label_train = data_utils.grouped_shuffle([raw_train, label_train])
    raw_val, label_val = data_utils.grouped_shuffle([raw_val, label_val])
    print('{}-Quick Draw data loaded!'.format(datetime.now()))
    sys.stdout.flush()

    return raw_train, label_train, raw_val, label_val
