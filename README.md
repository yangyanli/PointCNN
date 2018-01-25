# PointCNN

Created by <a href="http://yangyan.li" target="_blank">Yangyan Li</a>, Rui Bu, <a href="http://www.mcsun.cn" target="_blank">Mingchao Sun</a>, and <a href="http://www.cs.sdu.edu.cn/~baoquan/" target="_blank">Baoquan Chen</a> from Shandong University.

### Introduction

PointCNN is a simple and general framework for feature learning from point cloud, which refreshed five benchmark records in point cloud processing, including:

* classification accuracy on ModelNet40 (91.7%)
* classification accuracy on ScanNet (77.9%)
* segmentation part averaged IoU on ShapeNet Parts (86.13%)
* segmentation mean IoU on S3DIS (62.74%)
* per voxel labelling accuracy on ScanNet (85.1%). 

See our <a href="http://arxiv.org/abs/1801.07791" target="_blank">research paper on arXiv</a> for more details.

### Code Organization
The core X-Conv and PointCNN architecture are defined in ./pointcnn.py.

The network/training/data augmentation hyperparameters for classification tasks are defined in ./pointcnn_cls/\*.py, for segmentation tasks are defined in ./pointcnn_seg/\*.py

### Usage

Commands for training and testing ModelNet40 classification:
```
cd data_conversions
python3 ./download_datasets.py -d modelnet
cd ../pointcnn_cls
./train_val_modelnet.sh -g 0 -x modelnet_x2_l4
```

Commands for training and testing ShapeNet Parts segmentation:
```
cd data_conversions
python3 ./download_datasets.py -d shapenet_partseg
cd ../pointcnn_seg
./train_val_shapenet.sh -g 0 -x shapenet_x8_2048_fps
./test_shapenet.sh -g 0 -x shapenet_x8_2048_fps -l ../../models/seg/pointcnn_seg_shapenet_x8_2048_fps_xxxx/ckpts/iter-xxxxx -r 10
cd ..
python3 ./evaluate_seg.py -g ../data/shapenet_partseg/test_label -p ../data/shapenet_partseg/test_data_pred_10
```

Other datasets can be processed in a similar way.
