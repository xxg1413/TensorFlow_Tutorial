import  tensorflow as tf
tf.enable_eager_execution()

X = tf.constant([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]])
y = tf.constant([[10.0],[20.0]])


class LinearLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.w = self.add_variable(name="w", shape=[input_shape[-1], 1], initializer= tf.zeros_initializer())
        self.b = self.add_variable(name="b", shape=[1], initializer=tf.zeros_initializer())

    def call(self, X):
        y_pred = tf.matmul(X, self.w) + self.b
        return y_pred


class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer = LinearLayer()

    def call(self, input):
        output = self.layer(input)
        return output



model = Linear()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.reduce_mean(tf.square(y_pred - y))

    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars = zip(grads, model.variables))

print(model.variables)