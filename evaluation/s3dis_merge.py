#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import plyfile
import numpy as np
import argparse
import h5py

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafolder', '-d', help='Path to input *_pred.h5', required=True)
    args = parser.parse_args()
    print(args)


    categories_list = os.listdir(args.datafolder)

    for category in categories_list:
        output_path = os.path.join(args.datafolder,category,"pred.npy")
        label_length = np.load(os.path.join(args.datafolder,category,"label.npy")).shape[0]

        merged_label_zero = np.zeros((label_length),dtype=int)
        merged_confidence_zero = np.zeros((label_length),dtype=float)
        merged_label_half = np.zeros((label_length), dtype=int)
        merged_confidence_half = np.zeros((label_length), dtype=float)
        #merged_label = np.zeros((label_length,2))

        final_label = np.zeros((label_length), dtype=int)
        pred_list = [pred for pred in os.listdir(os.path.join(args.datafolder,category)) if pred.split(".")[-1] == "h5" and "pred" in pred]
        for pred_file in pred_list:
            print(os.path.join(args.datafolder,category, pred_file))
            data = h5py.File(os.path.join(args.datafolder,category, pred_file))
            labels_seg = data['label_seg'][...].astype(np.int64)
            indices = data['indices_split_to_full'][...].astype(np.int64)
            confidence = data['confidence'][...].astype(np.float32)
            data_num = data['data_num'][...].astype(np.int64)

            if 'zero' in pred_file:
                for i in range(labels_seg.shape[0]):
                    merged_label_zero[indices[i][:data_num[i]]] = labels_seg[i][:data_num[i]]
                    merged_confidence_zero[indices[i][:data_num[i]]] = confidence[i][:data_num[i]]
            else:
                for i in range(labels_seg.shape[0]):
                    merged_label_half[indices[i][:data_num[i]]] = labels_seg[i][:data_num[i]]
                    merged_confidence_half[indices[i][:data_num[i]]] = confidence[i][:data_num[i]]

        final_label[merged_confidence_zero >= merged_confidence_half] = merged_label_zero[merged_confidence_zero >= merged_confidence_half]
        final_label[merged_confidence_zero < merged_confidence_half] = merged_label_half[merged_confidence_zero < merged_confidence_half]

        np.savetxt(output_path,final_label,fmt='%d')
        print("saved to ",output_path)


if __name__ == '__main__':
    main()
