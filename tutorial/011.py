#!encoding=utf-8
'''
tensorflow save
'''

import  tensorflow as tf
import  numpy as np

#save

#注意 需要定义一样的dtype
W = tf.Variable([1,2,3],[3,4,5],dtype=tf.float32,name='weights')
b = tf.Variable([1,2,3],dtype=tf.float32,name='biases')

init = tf.initialize_all_variables()


saver = tf.train.Saver()

with tf.Session() as session:
    session.run(init)
    save_path = saver.save(session,'011/011.ckpt')
    print(save_path)



#load
#需要重新定义变量

W = tf.Variable( np.arange(6).reshape((2,3)),dtype=tf.float32,name='weights')
b = tf.Variable( np.arange(3).reshape((1,3)),dtype=tf.float32,name='biases')

saver = tf.train.Saver()
with tf.Session() as session:
    saver.restore(session, '011/011.ckpt')
    print(session.run(W))
    print(session.run(b))

