import tensorflow as tf
import custom_op
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import gradient_checker

class MyAddTest(tf.test.TestCase):
    """Unit Test for MyAdd"""
    
    def testMyAddFloat32(self):
        aa = np.random.rand(10).astype(np.float32)
        bb = np.random.rand(10).astype(np.float32)
        with self.test_session():
            result = custom_op.my_add(aa, bb)
            self.assertAllClose(result.eval(), aa+bb)

    def testMyAddFloat64(self):
        aa = np.random.rand(10).astype(np.float64)
        bb = np.random.rand(10).astype(np.float64)
        with self.test_session():
            result = custom_op.my_add(aa, bb)
            self.assertAllClose(result.eval(), aa+bb)

    def testMyAddGradFloat32(self):
        with self.test_session():
            x0 = constant_op.constant(
                  [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
                  shape=[2, 5],
                  name="x0")
            x1 = constant_op.constant(
                  [-0.8, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
                  shape=[2, 5],
                  name="x1")
            y = custom_op.my_add(x0, x1)
            x0_init = np.asarray(
                [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
                dtype=np.float32,
                order="F")
            x1_init = np.asarray(
                [[-0.8, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
                dtype=np.float32,
                order="F")
            err0 = gradient_checker.compute_gradient_error(
                x0, [2, 5], y, [2, 5], x_init_value=x0_init,
                extra_feed_dict={x1:x1_init})
            err1 = gradient_checker.compute_gradient_error(
                x1, [2, 5], y, [2, 5], x_init_value=x1_init,
                extra_feed_dict={x0:x0_init})
        print("my_add (float32) dy/dx0 err0 = ", err0)
        print("my_add (float32) dy/dx1 err1 = ", err1)
        self.assertLess(err0, 1e-4) 
        self.assertLess(err1, 1e-4) 

    def testMyAddGradFloat64(self):
        with self.test_session():
            x0 = constant_op.constant(
                  [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
                  shape=[2, 5],
                  dtype=tf.float64,
                  name="x0")
            x1 = constant_op.constant(
                  [-0.8, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
                  shape=[2, 5],
                  dtype=tf.float64,
                  name="x1")
            y = custom_op.my_add(x0, x1)
            x0_init = np.asarray(
                [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
                dtype=np.float64,
                order="F")
            x1_init = np.asarray(
                [[-0.8, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
                dtype=np.float64,
                order="F")
            err0 = gradient_checker.compute_gradient_error(
                x0, [2, 5], y, [2, 5], x_init_value=x0_init,
                extra_feed_dict={x1:x1_init})
            err1 = gradient_checker.compute_gradient_error(
                x1, [2, 5], y, [2, 5], x_init_value=x1_init,
                extra_feed_dict={x0:x0_init})
        print("my_add (float32) dy/dx0 err0 = ", err0)
        print("my_add (float32) dy/dx1 err1 = ", err1)
        self.assertLess(err0, 1e-10) 
        self.assertLess(err1, 1e-10) 

if __name__ == '__main__':
    tf.test.main()
