# /usr/local/cuda-11.4/bin/nvcc -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I /home/sinanb/anaconda3/envs/pointnet2/lib/python2.7/site-packages/tensorflow/include -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 && g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I /home/sinanb/anaconda3/envs/pointnet2/lib/python2.7/site-packages/tensorflow/include -L /usr/local/cuda-8.0/lib64 -O2

#!/usr/bin/env bash

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

/usr/local/cuda-11.4/bin/nvcc -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $TF_INC -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

# TF1.13
g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $TF_INC -I /usr/local/cuda-11.4/include -I $TF_INC/external/nsync/public -lcudart -L /usr/local/cuda-11.4/lib64/ -L$TF_LIB -ltensorflow_framework -O2 # -D_GLIBCXX_USE_CXX11_ABI=0
