#!encoding=utf-8
'''
tensorflow optimizer

https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html

可视化神经网络
tensorflow - tensorboard

'''


import  tensorflow as tf
import  numpy as np
import  matplotlib.pyplot as plot


#神经层函数
def add_layer(inputs, in_size,out_size, activtion_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')

        with tf.name_scope('biases'):
            biases = tf.Variable( tf.zeros([1,out_size]) + 0.1, name='b') #推荐不为0
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases

        if activtion_function  is None:
            outputs = Wx_plus_b
        else:
            outputs = activtion_function(Wx_plus_b)
        return  outputs


with tf.name_scope('input'):

    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')


l1 = add_layer(xs, 1, 10,activtion_function=tf.nn.relu)

predition = add_layer(l1,10,1,activtion_function=None)


with tf.name_scope('loss'):
    loss = tf.reduce_mean( tf.reduce_sum( tf.square(ys-predition),reduction_indices=[1] ), name='loss')

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


init = tf.initialize_all_variables()


session = tf.Session()

##加载到一个文件里面
writer = tf.train.SummaryWriter('006/',session.graph )

session.run(init)

'''
tensorboard --logdir='006'                                                                                      16:30:05  ☁  master ☂ ⚡ ✭
Starting TensorBoard b'23' on port 6006

(You can navigate to http://0.0.0.0:6006)

打开浏览器之后，在graph 菜单下
'''