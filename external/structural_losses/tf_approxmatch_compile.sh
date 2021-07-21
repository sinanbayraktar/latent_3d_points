# set -e
# if [ 'tf_approxmatch_g.cu.o' -ot 'tf_approxmatch_g.cu' ] ; then
# 	echo 'nvcc'
# 	/usr/local/cuda-11.4/bin/nvcc tf_approxmatch_g.cu -o tf_approxmatch_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# fi
# if [ 'tf_approxmatch_so.so'  -ot 'tf_approxmatch.cpp' ] || [ 'tf_approxmatch_so.so'  -ot 'tf_approxmatch_g.cu.o' ] ; then
# 	echo 'g++'
# 	g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I /home/sinanb/anaconda3/envs/pointnet2/lib/python2.7/site-packages/tensorflow/include -I /usr/local/cuda-8.0/include  -L /usr/local/cuda-8.0/lib64/ -O2
# fi

#!/usr/bin/env bash

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

/usr/local/cuda-11.4/bin/nvcc tf_approxmatch_g.cu -o tf_approxmatch_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.13
g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I $TF_INC -I /usr/local/cuda-11.4/include -I $TF_INC/external/nsync/public -lcudart -L /usr/local/cuda-11.4/lib64/ -L$TF_LIB -ltensorflow_framework -O2 # -D_GLIBCXX_USE_CXX11_ABI=0


