#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import plyfile
import numpy as np
import argparse
import h5py

reduced_length_dict = {"MarketplaceFeldkirch":[10538633,"marketsquarefeldkirch4-reduced"],
                       "StGallenCathedral":[14608690,"stgallencathedral6-reduced"],
                       "sg27":[28931322,"sg27_10-reduced"],
                       "sg28":[24620684,"sg28_2-reduced"]}

full_length_dict = {"stgallencathedral_station1":[31179769,"stgallencathedral1"],
                    "stgallencathedral_station3":[31643853,"stgallencathedral3"],
                    "stgallencathedral_station6":[32486227,"stgallencathedral6"],
                    "marketplacefeldkirch_station1":[26884140,"marketsquarefeldkirch1"],
                    "marketplacefeldkirch_station4":[23137668,"marketsquarefeldkirch4"],
                    "marketplacefeldkirch_station7":[23419114,"marketsquarefeldkirch7"],
                    "birdfountain_station1":[40133912,"birdfountain1"],
                    "castleblatten_station1":[31806225,"castleblatten1"],
                    "castleblatten_station5":[49152311,"castleblatten5"],
                    "sg27_station3":[422445052,"sg27_3"],
                    "sg27_station6":[226790878,"sg27_6"],
                    "sg27_station8":[429615314,"sg27_8"],
                    "sg27_station10":[285579196,"sg27_10"],
                    "sg28_station2":[170158281,"sg28_2"],
                    "sg28_station5":[267520082,"sg28_5"]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafolder', '-d', help='Path to input *_pred.h5', required=True)
    parser.add_argument('--version', '-v', help='full or reduced', type=str, required=True)
    args = parser.parse_args()
    print(args)

    if args.version == 'full':
        length_dict = full_length_dict
    else:
        length_dict = reduced_length_dict

    categories_list = [category for category in length_dict]
    print(categories_list)

    for category in categories_list:
        output_path = os.path.join(args.datafolder,"results",length_dict[category][1]+".labels")
        if not os.path.exists(os.path.join(args.datafolder,"results")):
            os.makedirs(os.path.join(args.datafolder,"results"))
        pred_list = [pred for pred in os.listdir(args.datafolder)
                     if category in pred  and pred.split(".")[0].split("_")[-1] == 'pred']

        label_length = length_dict[category][0]
        merged_label = np.zeros((label_length),dtype=int)
        merged_confidence = np.zeros((label_length),dtype=float)

        for pred_file in pred_list:
            print(os.path.join(args.datafolder, pred_file))
            data = h5py.File(os.path.join(args.datafolder, pred_file))
            labels_seg = data['label_seg'][...].astype(np.int64)
            indices = data['indices_split_to_full'][...].astype(np.int64)
            confidence = data['confidence'][...].astype(np.float32)
            data_num = data['data_num'][...].astype(np.int64)

            for i in range(labels_seg.shape[0]):
                temp_label = np.zeros((data_num[i]),dtype=int)
                pred_confidence = confidence[i][:data_num[i]]
                temp_confidence = merged_confidence[indices[i][:data_num[i]]]

                temp_label[temp_confidence >= pred_confidence] = merged_label[indices[i][:data_num[i]]][temp_confidence >= pred_confidence]
                temp_label[pred_confidence > temp_confidence] = labels_seg[i][:data_num[i]][pred_confidence > temp_confidence]

                merged_confidence[indices[i][:data_num[i]][pred_confidence > temp_confidence]] = pred_confidence[pred_confidence > temp_confidence]
                merged_label[indices[i][:data_num[i]]] = temp_label

        np.savetxt(output_path,merged_label+1,fmt='%d')

if __name__ == '__main__':
    main()
