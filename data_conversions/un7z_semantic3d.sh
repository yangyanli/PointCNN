#!/usr/bin/env bash -e
BASE_DIR=${1-../../data/semantic3d}

# Helper function to skip unpacking if already unpacked. Uses markers
# to indicate when a file is successfully unpacked.
unpack() {
    local path=${1}
    local marker=$path.unpacked
    if [ -e $marker ]; then
        echo "$path already unpacked, skipping"
        return
    fi
    7z x $path -o$(dirname $path) -y
    touch $marker
}

# Training data
unpack $BASE_DIR/train/bildstein_station1_xyz_intensity_rgb.7z
unpack $BASE_DIR/train/bildstein_station5_xyz_intensity_rgb.7z
unpack $BASE_DIR/train/domfountain_station1_xyz_intensity_rgb.7z
unpack $BASE_DIR/train/domfountain_station3_xyz_intensity_rgb.7z
unpack $BASE_DIR/train/neugasse_station1_xyz_intensity_rgb.7z
unpack $BASE_DIR/train/sg27_station1_intensity_rgb.7z
unpack $BASE_DIR/train/sg27_station2_intensity_rgb.7z
unpack $BASE_DIR/train/sg27_station5_intensity_rgb.7z
unpack $BASE_DIR/train/sg27_station9_intensity_rgb.7z
unpack $BASE_DIR/train/sg28_station4_intensity_rgb.7z
unpack $BASE_DIR/train/untermaederbrunnen_station1_xyz_intensity_rgb.7z
unpack $BASE_DIR/train/sem8_labels_training.7z

# Validation data
unpack $BASE_DIR/val/bildstein_station3_xyz_intensity_rgb.7z
unpack $BASE_DIR/val/domfountain_station2_xyz_intensity_rgb.7z
unpack $BASE_DIR/val/sg27_station4_intensity_rgb.7z
unpack $BASE_DIR/val/untermaederbrunnen_station3_xyz_intensity_rgb.7z

# Testing data
unpack $BASE_DIR/test/birdfountain_station1_xyz_intensity_rgb.7z
unpack $BASE_DIR/test/castleblatten_station1_intensity_rgb.7z
unpack $BASE_DIR/test/castleblatten_station5_xyz_intensity_rgb.7z
unpack $BASE_DIR/test/marketplacefeldkirch_station1_intensity_rgb.7z
unpack $BASE_DIR/test/marketplacefeldkirch_station4_intensity_rgb.7z
unpack $BASE_DIR/test/marketplacefeldkirch_station7_intensity_rgb.7z
unpack $BASE_DIR/test/sg27_station10_intensity_rgb.7z
unpack $BASE_DIR/test/sg27_station3_intensity_rgb.7z
unpack $BASE_DIR/test/sg27_station6_intensity_rgb.7z
unpack $BASE_DIR/test/sg27_station8_intensity_rgb.7z
unpack $BASE_DIR/test/sg28_station2_intensity_rgb.7z
unpack $BASE_DIR/test/sg28_station5_xyz_intensity_rgb.7z
unpack $BASE_DIR/test/stgallencathedral_station1_intensity_rgb.7z
unpack $BASE_DIR/test/stgallencathedral_station3_intensity_rgb.7z
unpack $BASE_DIR/test/stgallencathedral_station6_intensity_rgb.7z
