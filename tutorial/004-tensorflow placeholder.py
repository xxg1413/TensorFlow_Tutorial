#encoding=utf-8
'''
tensorflow placeholder
'''


import  tensorflow as tf


input1= tf.placeholder(tf.float32)
input2= tf.placeholder(tf.float32)


output = tf.mul(input1,input2)

with tf.Session() as sess:
    #feed_dict 传入参数
    print(sess.run(output, feed_dict={input1:[7.0], input2:[2.0]}))
