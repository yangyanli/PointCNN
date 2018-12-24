#/bin/bash
PYTHON=python3
CUDA_PATH=/usr/local/cuda
TF_LIB=$($PYTHON -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
PYTHON_VERSION=$($PYTHON -c 'import sys; print("%d.%d"%(sys.version_info[0], sys.version_info[1]))')
TF_PATH=$TF_LIB/include
$CUDA_PATH/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -L$TF_LIB -ltensorflow_framework -I $TF_PATH/external/nsync/public/ -I $TF_PATH -I $CUDA_PATH/include -lcudart -L $CUDA_PATH/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0
