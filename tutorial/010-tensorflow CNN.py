#!encoding=utf-8
'''
tensorflow CNN

https://www.tensorflow.org/versions/r0.10/tutorials/deep_cnn/index.html

'''


import  tensorflow as tf

from  tensorflow.examples.tutorials.mnist  import input_data


#加载数据文件
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)


xs = tf.placeholder(tf.float32,[None,784]) #784为每张图片的像素点总和 28*28
ys = tf.placeholder(tf.float32,[None,10]) #十个输出的值

keep_prob = tf.placeholder(tf.float32)


x_image = tf.reshape(xs,[-1,28,28,1])
#print(x_image.shape)




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

#准确度
def compute_accuracy(v_xs, v_ys):
    global  prediction
    y_pre = session.run(prediction, feed_dict={xs:v_xs}) #生成的预测值是概率值

    #是否正确预测
    corrent_prediction = tf.equal(tf.arg_max(y_pre,1), tf.argmax(v_ys, 1))

    #统计误差
    accuarcy = tf.reduce_mean(tf.cast(corrent_prediction, tf.float32))
    result = session.run(accuarcy,feed_dict={xs:v_xs, ys:v_ys})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


#CNN 层
def conv2d(x, W):
    return  tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME') #padding的两种形式：VALID SAME


#减小长和宽
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


#定义层

#conv1 layer
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(  conv2d(x_image, W_conv1) + b_conv1 ) # 28*28*32
h_pool1 = max_pool_2x2(h_conv1)   #14*14*32



#conv2 layer
W_conv2 = weight_variable([5,5,32,64]) #不断变高
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2 ) #14*14*64
h_pool2 = max_pool_2x2(h_conv2)  #7*7*64



##func1 layer
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])# 7,7,64 -->7*7*64

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


#func2 layer
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

prediction = tf.nn.softmax( tf.matmul(h_fc1_drop, W_fc2) + b_fc2)




prediction= add_layer(xs, 784, 10, activtion_function=tf.nn.softmax) #softmax 用来做分类

##计算误差
cross_entropy = tf.reduce_mean( -tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


session = tf.Session()

session.run(tf.initialize_all_variables())


for i in range(100000 ):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})

    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))



