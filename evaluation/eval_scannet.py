#!/usr/bin/python3
"""Merge blocks and evaluate scannet"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import plyfile
import numpy as np
import argparse
import h5py
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafolder', '-d', help='Path to input *_pred.h5', required=True)
    parser.add_argument('--picklefile', '-p', help='Path to scannet_test.pickle', required=True)
    args = parser.parse_args()
    print(args)

    file_list = os.listdir(args.datafolder)
    pred_list = [pred for pred in file_list if pred.split(".")[-1] == "h5" and "pred" in pred]

    pts_acc_list = []
    vox_acc_list = []

    #load scannet_test.pickle file
    file_pickle = open(args.picklefile, 'rb')
    xyz_all = pickle.load(file_pickle, encoding='latin1') # encoding keyword for python3
    labels_all = pickle.load(file_pickle, encoding='latin1')
    file_pickle.close()

    pickle_dict = {}
    for room_idx, xyz in enumerate(xyz_all):

        room_pt_num = xyz.shape[0]
        room_dict = {}

        room_dict["merged_label_zero"] = np.zeros((room_pt_num),dtype=int)
        room_dict["merged_confidence_zero"] = np.zeros((room_pt_num),dtype=float)
        room_dict["merged_label_half"] = np.zeros((room_pt_num), dtype=int)
        room_dict["merged_confidence_half"] = np.zeros((room_pt_num), dtype=float)
        room_dict["final_label"] = np.zeros((room_pt_num), dtype=int)

        pickle_dict[room_idx] = room_dict

    # load block preds and merge them to room scene
    for pred_file in pred_list:

        print("process:", os.path.join(args.datafolder, pred_file))
        test_file = pred_file.replace("_pred","")

        # load pred .h5
        data_pred = h5py.File(os.path.join(args.datafolder, pred_file))

        pred_labels_seg = data_pred['label_seg'][...].astype(np.int64)
        pred_indices = data_pred['indices_split_to_full'][...].astype(np.int64)
        pred_confidence = data_pred['confidence'][...].astype(np.float32)
        pred_data_num = data_pred['data_num'][...].astype(np.int64)

        
        if 'zero' in pred_file:
            for b_id in range(pred_labels_seg.shape[0]):
                indices_b = pred_indices[b_id]
                for p_id in range(pred_data_num[b_id]):
                    room_indices = indices_b[p_id][0]
                    inroom_indices = indices_b[p_id][1]
                    pickle_dict[room_indices]["merged_label_zero"][inroom_indices] = pred_labels_seg[b_id][p_id]
                    pickle_dict[room_indices]["merged_confidence_zero"][inroom_indices] = pred_confidence[b_id][p_id]
        else:
            for b_id in range(pred_labels_seg.shape[0]):
                indices_b = pred_indices[b_id]
                for p_id in range(pred_data_num[b_id]):
                    room_indices = indices_b[p_id][0]
                    inroom_indices = indices_b[p_id][1]
                    pickle_dict[room_indices]["merged_label_half"][inroom_indices] = pred_labels_seg[b_id][p_id]
                    pickle_dict[room_indices]["merged_confidence_half"][inroom_indices] = pred_confidence[b_id][p_id]

    for room_id in pickle_dict.keys():

        final_label = pickle_dict[room_id]["final_label"]
        merged_label_zero = pickle_dict[room_id]["merged_label_zero"]
        merged_label_half = pickle_dict[room_id]["merged_label_half"]
        merged_confidence_zero = pickle_dict[room_id]["merged_confidence_zero"]
        merged_confidence_half = pickle_dict[room_id]["merged_confidence_half"]

        final_label[merged_confidence_zero >= merged_confidence_half] = merged_label_zero[merged_confidence_zero >= merged_confidence_half]
        final_label[merged_confidence_zero < merged_confidence_half] = merged_label_half[merged_confidence_zero < merged_confidence_half]

    # eval
    for room_id, pts in enumerate(xyz_all):

        label = labels_all[room_id]
        pred = pickle_dict[room_id]["final_label"]
        data_num = pts.shape[0]

        # compute pts acc (ignore label 0 which is scannet unannotated)
        c_accpts = np.sum(np.equal(pred,label))
        c_ignore = np.sum(np.equal(label,0))
        pts_acc_list.append([c_accpts, data_num - c_ignore])

        # compute voxel accuracy (follow scannet and pointnet++)
        res = 0.0484
        coordmax = np.max(pts, axis=0)
        coordmin = np.min(pts, axis=0)
        nvox = np.ceil((coordmax - coordmin) / res)
        vidx = np.ceil((pts - coordmin) / res)
        vidx = vidx[:, 0] + vidx[:, 1] * nvox[0] + vidx[:, 2] * nvox[0] * nvox[1]
        uvidx, vpidx = np.unique(vidx, return_index=True)

        # compute voxel label
        uvlabel = np.array(label)[vpidx]

        # compute voxel pred (follow pointnet++ majority voting)
        uvpred_tp = []
        label_pred_dict = {}

        for uidx in uvidx:
            label_pred_dict[int(uidx)] = []
        for k, p in enumerate(pred):
            label_pred_dict[int(vidx[k])].append(p)
        for uidx in uvidx:
            uvpred_tp.append(np.argmax(np.bincount(label_pred_dict[int(uidx)])))

        # compute voxel accuracy (ignore label 0 which is scannet unannotated)
        c_accvox = np.sum(np.equal(uvpred_tp, uvlabel))
        c_ignore = np.sum(np.equal(uvlabel,0))

        vox_acc_list.append([c_accvox, (len(uvlabel) - c_ignore)])

    # compute avg pts acc
    pts_acc_sum = np.sum(pts_acc_list,0)
    print("pts acc", pts_acc_sum[0]*1.0/pts_acc_sum[1])

    #compute avg voxel acc
    vox_acc_sum = np.sum(vox_acc_list,0)
    print("voxel acc", vox_acc_sum[0]*1.0/vox_acc_sum[1])

if __name__ == '__main__':
    main()
