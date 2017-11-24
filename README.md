---
title: Custom Ops for Tensorflow
date: 21-11-2017
note: Verified working with r1.4
---

# Introduction

A simple custom operation written in C++ for Tensorflow. Meant to be used as
a self contained example and a template for actual projects.

This method does not require building Tensorflow from source.

# Files

- `custom_op.cc` - C++ implementation of operation
- `custom_op.py` - Python wrapper and gradient implementation for op
- `custom_op_test.py` - Unit test for custom op
- `Makefile` - Build rules
- `README.md` - This file

# Implementation of the operation

A simple add operation, $y = x0 + x1$, which takes 2 inputs is implemented in
`custom_op.cc`. Polymorphism to support multiple input types using C++
templates is also demonstrated.

The pip package contains necessary header files for building custom ops. 
The location of the headers can be found using 
`python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())'`

# Creating the Python Wrapper

Tensorflow provides a `tf.load_op_library()` to load custom ops. A very basic
python wrapper is also created. CamelCase operations defined in C++ are are
converted to lower case and underscores naming convention. 

# Defining the gradient

Gradients of custom operations needs to be defined for automatic
differentiation to work. The `@ops.RegisterGradient()` decorator allows ops to
be registered as gradients. In this example, the gradient is implemented in
python. It could also be implemented in C++ and registered similarly.

# Unit tests

Tensorflow provides some framework for testing custom ops and checking the
gradients using finite differences. `custom_op_test.py` provides a simple
example on how to use these functionalities


# Build Notes

Tensorflow packages installed using pip was built using gcc4, if using the pip
installed version and compiling this with gcc>=5, add
`-D_GLIBCXX_USE_CXX11_ABI=0` to CFLAGS in `Makefile`. If you compiled your own
tensorflow, and you run into some runtime error, adding or removing this flag
should fix your problem.

