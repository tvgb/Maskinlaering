import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def get_training_data(num):
    x_train = mnist.train.images[:num, :]
    y_train = mnist.train.labels[:num, :]
    return x_train, y_train


def get_test_data(num):
    x_test = mnist.test.images[:num,:]
    y_test = mnist.test.labels[:num,:]
    return x_test, y_test


class MinstModel:
    def __init__(self):

        # MAKING PLACEHOLDERS FOR THE INPUT DATA
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

        # WEIGHT AND BIAS
        self.W = tf.Variable(tf.zeros([784, 10]))
        self.b = tf.Variable(tf.zeros([10]))

        # PREDICTOR
        self.logits = tf.matmul(self.x, self.W) + self.b
        self.f = tf.nn.softmax(self.logits)

        # CALCULATING LOSS OR CROSS ENTROPY
        self.loss = tf.losses.softmax_cross_entropy(self.y, self.f)


model = MinstModel()

# DECLARING THE SESSION
sess = tf.Session()

# STEPS AND LEARNING RATE
LEARNING_RATE = 4.1
TRAIN_STEPS = 500

# GETTING DATA
x_train, y_train = get_training_data(60000)
x_test, y_test = get_test_data(10000)

init = tf.global_variables_initializer()

# RUNNING SESSION
sess.run(init)

training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(model.loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(model.f, 1), tf.argmax(model.y, 1)), tf.float32))

# TRAINING
for epoch in range(TRAIN_STEPS+1):
    frac = epoch / TRAIN_STEPS
    filled_progbar = round(frac * 20)
    print('\r', '#' * filled_progbar + '-' * (20 - filled_progbar), '[{:>7.2%}]'.format(frac), end='')
    sys.stdout.flush()
    sess.run(training, feed_dict={model.x: x_train, model.y: y_train})

print('\nTraining Step: ' + str(TRAIN_STEPS) + '  Accuracy = ' + str(
    sess.run(accuracy, feed_dict={model.x: x_test, model.y: y_test})) + '  Loss = ' + str(
    sess.run(model.loss, {model.x: x_train, model.y: y_train})))

# VISUALIZING W
for i in range(10):
    plt.subplot(2, 5, i+1)
    weight = sess.run(model.W)[:, i]
    plt.title(i)
    plt.imshow(weight.reshape([28, 28]), cmap=plt.get_cmap('RdBu'))
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)


plt.show()
