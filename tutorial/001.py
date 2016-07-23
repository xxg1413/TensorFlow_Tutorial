#!encoding=utf-8
'''
基本的tensorflow 流程
'''

import tensorflow as tf
import numpy as np

#产生随机的数据
X_data  = np.random.rand(100).astype(np.float32)
y_data  = X_data*0.1 + 0.3


#创建tensorflow结构

#随机[-1,1]中的数
Weigts = tf.Variable(tf.random_uniform([1], -1.0,1.0))

#0
biases = tf.Variable(tf.zeros([1]))

y = Weigts * X_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))

#优化器
optimizer = tf.train.GradientDescentOptimizer(0.5)

train = optimizer.minimize(loss)


#初始化
init = tf.initialize_all_variables()



sess = tf.Session()
sess.run(init)  #激活init


for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weigts), sess.run(biases))
