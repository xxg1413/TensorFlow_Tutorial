#encoding=utf-8
'''
tensorflow 变量

'''


import  tensorflow as tf

#定义一个变量 名字为counter
state = tf.Variable(0,name="counter")

print(state.name)

#常量
one = tf.constant(1)

new_value = tf.add(state,one)

update = tf.assign(state, new_value)


#初始化所有的变量
init = tf.initialize_all_variables()

with tf.Session() as session:
    #这里 初始化变量
    session.run(init)

    for _ in range(3):
        session.run(update)
        #这里不能直接填state
        print(session.run(state))