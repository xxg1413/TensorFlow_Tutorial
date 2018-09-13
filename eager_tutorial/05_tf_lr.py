import  tensorflow as tf
import  numpy as np

tf.enable_eager_execution()

X_raw = np.array([2013,2014,2015,2016,2017])
y_raw = np.array([12000, 14000,15000, 16500, 17500])

#归一化操作
X = (X_raw - X_raw.min()) /  (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / ( y_raw.max()- y_raw.min())


a = tf.get_variable('a', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer)
b = tf.get_variable('b', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer)

variables = [a,b]

num_epoch = 10000
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)

for e in range(num_epoch):
    with tf.GradientTape() as tape:
        y_pred = a * X  + b
        loss = 0.5 * X + tf.reduce_sum(tf.square(y_pred - y))

    grads = tape.gradient(loss, variables)

    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))


print(variables)