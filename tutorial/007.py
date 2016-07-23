#!encoding=utf-8
'''
tensorflow tensorboard
'''


import  tensorflow as tf
import  numpy as np
import  matplotlib.pyplot as plot



#神经层函数
def add_layer(inputs, in_size,out_size, n_layer, activtion_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
            tf.histogram_summary(layer_name+'/weights',Weights)

        with tf.name_scope('biases'):
            biases = tf.Variable( tf.zeros([1,out_size]) + 0.1, name='b') #推荐不为0
            tf.histogram_summary(layer_name+'/biases',biases)

        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases


        if activtion_function  is None:
            outputs = Wx_plus_b
        else:
            outputs = activtion_function(Wx_plus_b)

        tf.histogram_summary(layer_name + '/outputs', outputs)

        return  outputs



x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise =  np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise


with tf.name_scope('input'):

    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')



l1 = add_layer(xs, 1, 10,n_layer=1, activtion_function=tf.nn.relu)


predition = add_layer(l1,10,1,n_layer=2, activtion_function=None)


with tf.name_scope('loass'):
    loss = tf.reduce_mean( tf.reduce_sum( tf.square(ys-predition),reduction_indices=[1] ), name='loss')

    #存量变化 在envet上显示
    tf.scalar_summary('loss',loss)


with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


init = tf.initialize_all_variables()

session = tf.Session()

merged = tf.merge_all_summaries()

writer = tf.train.SummaryWriter('007/',session.graph)


session.run(init)

for i in range(1000):
    session.run(train_step,feed_dict={xs:x_data, ys:y_data})
    if i % 50 == 0:
        result = session.run(merged, feed_dict={xs:x_data, ys: y_data})
        writer.add_summary(result,i)
