import numpy as np
import tensorflow as tf
import tqdm


(x_train_, y_train_), (x_test_, y_test_) = tf.keras.datasets.mnist.load_data()

x_train = np.reshape(x_train_, (-1, 28, 28, 1))  # tf.nn.conv2d takes 4D arguments
y_train = np.zeros((y_train_.size, 10))
y_train[np.arange(y_train_.size), y_train_] = 1

# x_train = np.true_divide(x_train, 255) # Reducing all numbers to a value between 0 and 1

train_batches = 500  # Divide training data into batches to speed up optimization
x_train_batches = np.split(x_train, train_batches)
y_train_batches = np.split(y_train, train_batches)

x_test = np.reshape(x_test_, (-1, 28, 28, 1))
y_test = np.zeros((y_test_.size, 10))
y_test[np.arange(y_test_.size), y_test_] = 1

test_batches = 10
x_test_batches = np.split(x_test, test_batches)
y_test_batches = np.split(y_test, test_batches)



class ConvolutionalNeuralNetworkModel:
    def __init__(self):

        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        W1 = tf.Variable(tf.random_normal([5, 5, 1, 32]))  # tf.nn.conv2d takes 4D arguments. 5x5 filters, 1 in-channel, 32 out-channels
        b1 = tf.Variable(tf.random_normal([32]))

        W2 = tf.Variable(tf.random_normal([3, 3, 32, 64]))  # (width / 2) * (height / 2) * 32. Divided by 2 due to pooling
        b2 = tf.Variable(tf.random_normal([64]))

        W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 10]))
        b3 = tf.Variable(tf.random_normal([10]))

        # Model1 operations
        conv1 = tf.nn.bias_add(tf.nn.conv2d(self.x, W1, strides=[1, 1, 1, 1], padding='SAME'), b1)  # Using builtin function for adding bias
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Model1 operations
        conv2 = tf.nn.bias_add(tf.nn.conv2d(pool1, W2, strides=[1, 1, 1, 1], padding='SAME'), b2)  # Using builtin function for adding bias
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Logits
        logits = tf.nn.bias_add(tf.matmul(tf.reshape(pool2, [-1, 7 * 7 * 64]), W3), b3)

        # Predictor
        f = tf.nn.softmax(logits)

        # Cross Entropy loss
        self.loss = tf.losses.softmax_cross_entropy(self.y, logits)

        # Accuracy
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(f, 1), tf.argmax(self.y, 1)), tf.float32))


model = ConvolutionalNeuralNetworkModel()

LEARNING_RATE = 0.005
EPOCH = 100


# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.AdamOptimizer(LEARNING_RATE).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

#session = tf.Session(config=tf.ConfigProto(log_device_placement=True))


# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

acc_table = []

for epoch in range(EPOCH):

    accuracy = 0

    # Running batches of training data to avoid OOM
    for batch in tqdm.trange(train_batches, desc=str(epoch + 1)+'/'+str(EPOCH)):
        session.run(minimize_operation, {model.x: x_train_batches[batch], model.y: y_train_batches[batch]})

    # Running batches of test data to avoid OOM
    for batch in range(test_batches):
        accuracy_string = ("epoch: " + str(epoch + 1) + "  Accuracy: ")
        accuracy += session.run(model.accuracy, {model.x: x_test_batches[batch], model.y: y_test_batches[batch]})

    acc_table.append(accuracy_string + str(round((accuracy / test_batches), 4)))

for i in range(len(acc_table)):
    print(acc_table[i])


session.close()
