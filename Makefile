CC=g++
CFLAGS=-Wall -std=c++11 -shared -fPIC #-D_GLIBCXX_USE_CXX11_ABI=0 #-fopenmp

#Output library name
TFOP=custom_op_impl
SOURCES=$(wildcard *.cc)

#Get location of Tensorflow headers and library files
TF_INC=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

TF_CFLAGS=-I$(TF_INC) -I$(TF_INC)/external/nsync/public -L$(TF_LIB) -ltensorflow_framework

all: $(TFOP).so

$(TFOP).so:$(SOURCES)
	$(CC) $(CFLAGS) $^ $(TF_CFLAGS) -o $@

clean:
	rm -f $(TFOP).so
