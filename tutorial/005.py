#!encoding=utf-8
'''
tensorflow 激励函数
列表：
https://www.tensorflow.org/versions/r0.10/api_docs/python/nn.html

如何添加一个神经层


'''

import  tensorflow as tf
import  numpy as np
import  matplotlib.pyplot as plot




#神经层函数

def add_layer(inputs, in_size,out_size, activtion_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])) + 0.1 #推荐不为0
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activtion_function  is None:
        outputs = Wx_plus_b

    else:
        outputs = activtion_function(Wx_plus_b)
    return  outputs

x_data = np.linspace(-1,1,300)[:,np.newaxis]

noise =  np.random.normal(0,0.05,x_data.shape)

print(noise)

y_data = np.square(x_data) - 0.5 + noise


xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])



l1 = add_layer(xs, 1, 10,activtion_function=tf.nn.relu)

#输出
predition = add_layer(l1,10,1,activtion_function=None)


#平均误差
loss = tf.reduce_mean( tf.reduce_sum( tf.square(ys-predition),reduction_indices=[1] ) )

#减少误差
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


init = tf.initialize_all_variables()


session = tf.Session()
session.run(init)

fig = plot.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)

plot.ion()
plot.show()


for i in range(1000):
    session.run(train_step,feed_dict={xs:x_data, ys:y_data})
    if i % 50 == 0:
        #print(session.run(loss, feed_dict={xs:x_data,ys:y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass

        predition_value = session.run(predition, feed_dict={xs:x_data})
        lines = ax.plot(x_data,predition_value,'r-',lw=5)

        plot.pause(0.1)
