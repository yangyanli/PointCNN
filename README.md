# PointCNN

Created by <a href="http://yangyan.li" target="_blank">Yangyan Li</a>,<a href="http://rbruibu.cn" target="_blank"> Rui Bu</a>, <a href="http://www.mcsun.cn" target="_blank">Mingchao Sun</a>, and <a href="http://www.cs.sdu.edu.cn/~baoquan/" target="_blank">Baoquan Chen</a> from Shandong University.

## Introduction

PointCNN is a simple and general framework for feature learning from point cloud, which refreshed five benchmark records in point cloud processing, including:

* classification accuracy on ModelNet40 (91.7%)
* classification accuracy on ScanNet (77.9%)
* segmentation part averaged IoU on ShapeNet Parts (86.13%)
* segmentation mean IoU on S3DIS (62.74%)
* per voxel labelling accuracy on ScanNet (85.1%).

See our <a href="http://arxiv.org/abs/1801.07791" target="_blank">research paper on arXiv</a> for more details.

## Code Organization
The core X-Conv and PointCNN architecture are defined in ./pointcnn.py.

The network/training/data augmentation hyper parameters for classification tasks are defined in ./pointcnn_cls/\*.py, for segmentation tasks are defined in ./pointcnn_seg/\*.py

## Usage

Here we list the commands for training/evaluating PointCNN on multiple datasets and tasks.

* ### Classification

  * #### ModelNet40
  ```
  cd data_conversions
  python3 ./download_datasets.py -d modelnet
  cd ../pointcnn_cls
  ./train_val_modelnet.sh -g 0 -x modelnet_x2_l4
  ```

  * #### tu_berlin
  ```
  cd data_conversions
  python3 ./download_datasets.py -d tu_berlin
  cd ../pointcnn_cls
  ./train_val_tu_berlin.sh -g 0 -x tu_berlin_x2_l5
  ```

  * #### MNIST
  ```
  cd data_conversions
  python3 ./download_datasets.py -d mnist
  cd ../pointcnn_cls
  ./train_val_mnist.sh -g 0 -x mnist_x2_l5
	```

  * #### CIFAR-10
  ```
  cd data_conversions
  python3 ./download_datasets.py -d cifar10
  cd ../pointcnn_cls
  ./train_val_cifar10.sh -g 0 -x cifar10_x2_l4
  ```

  * #### quick_darw
  ```
  cd data_conversions
  python3 ./download_datasets.py -d quick_draw
  cd ../pointcnn_cls
  ./train_val_quick_draw.sh -g 0 -x quick_draw_full_x2_l6
  ```

* ### Segmentation

  We use farthest point sampling (the implementation from <a href="https://github.com/charlesq34/pointnet2" target="_blank">PointNet++</a> in segmentation tasks. Compile FPS before the training/evaluation:
  ```
  cd sampling
  bash tf_sampling_compile.sh
  ```

  * #### ShapeNet
  ```
  cd data_conversions
  python3 ./download_datasets.py -d shapenet_partseg
  cd ../pointcnn_seg
  ./train_val_shapenet.sh -g 0 -x shapenet_x8_2048_fps
  ./test_shapenet.sh -g 0 -x shapenet_x8_2048_fps -l ../../models/seg/pointcnn_seg_shapenet_x8_2048_fps_xxxx/ckpts/iter-xxxxx -r 10
  cd ../evaluation
  python3 eval_shapenet_seg.py -g ../../data/shapenet_partseg/test_label -p ../../data/shapenet_partseg/test_data_pred_10
  ```

  * #### S3DIS
  Please refer to [data_conversions](data_conversions/README.md) for downloading S3DIS, then:
  ```
  cd data_conversions/split_data
  python3 s3dis_prepare_label.py
  python3 s3dis_split.py
  cd ..
  python3 prepare_multiChannel_seg_data.py -f ../../data/S3DIS/out_part_rgb/ -c 6
  mv S3DIS_files/* ../../data/S3DIS/out_part_rgb/
  ./train_val_s3dis.sh -g 0 -x s3dis_x8_2048_fps_k16
  ./test_s3dis.sh -g 0 -x s3dis_x8_2048_fps_k16 -l ../../models/seg/s3dis_x8_2048_fps_k16_xxxx/ckpts/iter-xxxxx -r 4
  cd ../evaluation
  python3 s3dis_upsampling.py
  python3 eval_s3dis.py
  ```
  Please notice that these command just for Area1 validation, after modify the train val path in train_val_s3dis.sh, test_s3dis.sh and s3dis_upsampling, you can get other Area results.

  * #### ScanNet
  Please refer to [data_conversions](data_conversions/README.md) for downloading ScanNet, then:
  ```
  cd data_conversions/split_data
  python3 scannet_split.py
  cd ..
  python3 prepare_multiChannel_seg_data.py -f ../../data/scannet/scannet_split_dataset/
  cd ../pointcnn_seg
  ./train_val_scannet.sh -g 0 -x scannet_x8_2048_k8_fps
  ./test_scannet.sh -g 0 -x scannet_x8_2048_k8_fps -l ../../models/seg/pointcnn_seg_scannet_x8_2048_k8_fps_xxxx/ckpts/iter-xxxxx -r 4
  cd ../evaluation
  python3 eval_scannet.py
  ```

  * #### Semantic3D
  ```
  cd data_conversions
  bash download_semantic3d.sh
  bash un7z_semantic3d.sh
  mkdir ../../data/semantic3d/val
  mv ../../data/semantic3d/train/bildstein_station3_xyz_intensity_rgb.txt ../../data/semantic3d/train/domfountain_station2_xyz_intensity_rgb.txt ../../data/semantic3d/train/sg27_station4_intensity_rgb.txt ../../data/semantic3d/train/untermaederbrunnen_station3_xyz_intensity_rgb.txt ../../data/semantic3d/val
  cd split_data
  python3 semantic3d
  cd ../pointcnn_seg
  ./train_val_semantic3d.sh -g 0 -x semantic3d_x8_2048_k16
  ```