import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

_op_impl = tf.load_op_library('./custom_op_impl.so')

def my_add(x, y):
    return _op_impl.my_add(x, y)

@ops.RegisterGradient('MyAdd')
def _my_add_grad(op, grad):
    # y = x0 + x1
    # dy/dx0 = 1
    # dy/dx1 = 1
    # dx0 = grad
    # dx1 = grad
    return [grad, grad]
