import  tensorflow as tf
import  numpy as np
tf.enable_eager_execution()


class DataLoader():
    def __init__(self):
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        self.train_data = mnist.train.images
        self.train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        self.eval_data = mnist.test.images
        self.eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)


    def get_bath(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index,:], self.train_labels[index]



class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

    def predict(self, inputs):
        logits = self(inputs)
        return tf.argmax(logits, axis=-1)



num_batches= 10000
batch_size = 50
learning_rate = 0.001


model = MLP()
data_loader = DataLoader()
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)


for batch_index in range(num_batches):
    X,y = data_loader.get_bath(batch_size)
    with tf.GradientTape() as tape:
        y_logit_pred = model(tf.convert_to_tensor(X))
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_logit_pred)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))

    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))



num_eval_samples = np.shape( data_loader.eval_labels)[0]
y_pred = model.predict(data_loader.eval_data).numpy()

print("test accuracy: %f" %(sum(y_pred == data_loader.eval_labels) / num_eval_samples))

