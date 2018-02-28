# Datasets Preparation

## Download

If you want to download :

* tu_berlin
* modelnet
* shapenet_partseg
* mnist
* cifar10
* quick_draw

You can use download_datasets.py:

```
python download_datasets.py -f [path to data folder] -d [Dataset to download]
```

For Scannet, Original dataset website: http://www.scan-net.org/ . We follow [pointnet++ preprocessed data](https://github.com/charlesq34/pointnet2/tree/master/scannet) ([Onedrive link](https://1drv.ms/u/s!ApbTjxa06z9CgQhxDuSJPB5-FHtm)).

For S3DIS, you can download it at the original dataset website: http://buildingparser.stanford.edu/dataset.html#Download

## Convert to .h5 Files

**[optional]** For big scene point cloud datasets like **Scannet** and **S3DIS**, it's better to split into small blocks for training:

```
python3 scannet_split.py
```

```
python3 S3DIS_prepare_label.py
python3 S3DIS_split.py
```

After above manipulates, you can use this command to generate .h5 files:

```
python3 prepare_[dataset]_data.py -f [Path to data folder]
```

If you want to use extra features, such as RGB, you can use:

```
python3 prepare_multiChannel_seg_data.py -f [Path to data folder] -c [Channel number]
```

