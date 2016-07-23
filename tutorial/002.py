#encoding=utf-8

'''
tensorflow 会话控制
'''


import  tensorflow as tf


#矩阵乘法

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])

product = tf.matmul(matrix1, matrix2)

#方法1

session1 = tf.Session()
result = session1.run(product)
print(result)

session1.close()


#方法2
with tf.Session() as session2:
    result = session2.run(product)
    print(result)


