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



mode="train"
num_batches= 10000
batch_size = 50
learning_rate = 0.001

data_loader = DataLoader()

def train():

    model = MLP()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    checkoutpoint = tf.train.Checkpoint(myModel=model)

    summary_writer = tf.contrib.summary.create_file_writer("./log")
    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        for batch_index in range(num_batches):
            X,y = data_loader.get_bath(batch_size)
            with tf.GradientTape() as tape:
                y_logit_pred = model(tf.convert_to_tensor(X))
                loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_logit_pred)
                print("batch %d: loss %f" % (batch_index, loss.numpy()))
                tf.contrib.summary.scalar("loss",loss, step=batch_index)


            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

            if (batch_index + 1) % 100 ==0:
                checkoutpoint.save("./save/model.ckpt")



def test():

    restored_model = MLP()
    checkpoint = tf.train.Checkpoint(myModel=restored_model)
    checkpoint.restore(tf.train.latest_checkpoint("./save"))

    num_eval_samples = np.shape( data_loader.eval_labels)[0]
    y_pred = restored_model.predict(tf.constant(data_loader.eval_data)).numpy()

    print("test accuracy: %f" %(sum(y_pred == data_loader.eval_labels) / num_eval_samples))




if __name__ == "__main__":
    if mode == "train":
        train()
    if mode == "test":
        test()