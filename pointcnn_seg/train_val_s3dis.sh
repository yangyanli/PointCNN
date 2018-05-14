#!/usr/bin/env bash

gpu=
setting=
area=
models_folder="../../models/seg/"
data_folder="../../data/s3dis/"

usage() { echo "train/val pointcnn_seg with -g gpu_id -x setting -a area options"; }

gpu_flag=0
setting_flag=0
area_flag=0
while getopts g:x:a:h opt; do
  case $opt in
  g)
    gpu_flag=1;
    gpu=$(($OPTARG))
    ;;
  x)
    setting_flag=1;
    setting=${OPTARG}
    ;;
  a)
    area_flag=1;
    area=$(($OPTARG))
    ;;
  h)
    usage; exit;;
  esac
done

shift $((OPTIND-1))

if [ $gpu_flag -eq 0 ]
then
  echo "-g option is not presented!"
  usage; exit;
fi

if [ $setting_flag -eq 0 ]
then
  echo "-x option is not presented!"
  usage; exit;
fi

if [ $area_flag -eq 0 ]
then
  echo "-a option is not presented!"
  usage; exit;
fi

if [ ! -d "$models_folder" ]
then
  mkdir -p "$models_folder"
fi

echo "Train/Val with setting $setting on GPU $gpu for Area $area!"
CUDA_VISIBLE_DEVICES=$gpu python3 ../train_val_seg.py -t $data_folder/train_files_for_val_on_Area_$area.txt -v $data_folder/val_files_Area_$area.txt -s $models_folder -m pointcnn_seg -x $setting > $models_folder/pointcnn_seg_$setting.txt 2>&1 &
