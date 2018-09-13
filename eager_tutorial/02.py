import  tensorflow as tf
tf.enable_eager_execution()


x = tf.get_variable('x', shape=[1], initializer=tf.constant_initializer(3.))


with tf.GradientTape() as tape:
    y = tf.square(x)

y_grad = tape.gradient(y,x)
print([y.numpy(), y_grad.numpy()])

