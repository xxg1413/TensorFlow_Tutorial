#!encoding=utf-8
'''
tensorflow classification
'''

import  tensorflow as tf

from   tensorflow.examples.tutorials.mnist  import input_data


#加载数据文件
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)



xs = tf.placeholder(tf.float32,[None,784]) #784为每张图片的像素点总和 28*28
ys = tf.placeholder(tf.float32,[None,10]) #十个输出的值


#增加一个层
def add_layer(inputs, in_size,out_size, activtion_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])) + 0.1 #推荐不为0
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activtion_function  is None:
        outputs = Wx_plus_b

    else:
        outputs = activtion_function(Wx_plus_b)
    return  outputs


def compute_accuracy(v_xs, v_ys):
    global  prediction
    y_pre = session.run(prediction, feed_dict={xs:v_xs}) #生成的预测值是概率值

    #是否正确预测
    corrent_prediction = tf.equal(tf.arg_max(y_pre,1), tf.argmax(v_ys, 1))

    #统计误差
    accuarcy = tf.reduce_mean(tf.cast(corrent_prediction, tf.float32))
    result = session.run(accuarcy,feed_dict={xs:v_xs, ys:v_ys})
    return result



prediction= add_layer(xs, 784, 10, activtion_function=tf.nn.softmax) #softmax 用来做分类


##计算误差
cross_entropy = tf.reduce_mean( -tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


session = tf.Session()

session.run(tf.initialize_all_variables())


for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})

    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))



