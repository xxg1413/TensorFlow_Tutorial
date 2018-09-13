import tensorflow as tf
tf.enable_eager_execution()


A = tf.constant([[1,2],[3,4]])
B = tf.constant([[5,6],[7,8]])
C = tf.matmul(A,B)

print(C)