#!encoding=utf-8
'''
tensorflow overfitting

用dropout解决过拟合的问题
'''

import  tensorflow as tf
from sklearn.datasets import  load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing  import  LabelBinarizer



digits = load_digits()
X = digits.data
y = digits.target

#输出
y = LabelBinarizer().fit_transform(y)

X_train,X_test, y_train,y_test = train_test_split(X, y,test_size=0.3)


def add_layer(inputs, in_size,out_size, n_layer, activtion_function=None):
    layer_name = 'layer%s' % n_layer
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])) + 0.1 #推荐不为0

    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    #add drop
    Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_prob)

    if activtion_function  is None:
        outputs = Wx_plus_b

    else:
        outputs = activtion_function(Wx_plus_b)


    tf.histogram_summary(layer_name + '/outputs', outputs)

    return  outputs



keep_prob = tf.placeholder(tf.float32)


xs = tf.placeholder(tf.float32,[None,64])#64像素
ys = tf.placeholder(tf.float32,[None,10]) #十个输出的值



l1 = add_layer(xs,64,50,1,activtion_function=tf.nn.tanh)
prediction = add_layer(l1,50,10,2,activtion_function=tf.nn.softmax)


 ##计算误差
cross_entropy = tf.reduce_mean( -tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))

tf.scalar_summary('loss',cross_entropy)


train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)


session = tf.Session()


merged = tf.merge_all_summaries()

train_writer = tf.train.SummaryWriter('009/train',session.graph)
test_writer = tf.train.SummaryWriter('009/test',session.graph)


session.run(tf.initialize_all_variables())

for i in range(500):
    session.run(train_step, feed_dict={xs:X_train,ys:y_train,keep_prob: 0.5}) #保持50%不被drop
    if i % 50 == 0:
        train_result = session.run(merged, feed_dict={xs:X_train, ys:y_train,keep_prob: 1})
        test_result = session.run(merged, feed_dict={xs:X_test, ys:y_test, keep_prob: 1})
        train_writer.add_summary(train_result,i)
        test_writer.add_summary(test_result,i)


