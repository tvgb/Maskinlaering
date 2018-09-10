import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


x_train, y_train = [[0], [1]], [[1], [0]]

plt.plot(x_train, y_train, 'ro')


def graph(formula, x_range):
    x = np.array(x_range)
    y = eval(formula)
    plt.plot(x, y)
    plt.xlabel('x-akse')
    plt.ylabel('y-akse')
    plt.show()


class LinearRegressionModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable([[0.0]])
        self.b = tf.Variable([[0.0]])

        # Predictor
        f = tf.matmul(self.x, self.W) + self.b

        # Uses Mean Squared Error,
        # although instead of mean, sum is used.
        # self.loss = tf.reduce_sum(tf.square(f - self.y))
        self.loss = tf.losses.mean_squared_error(self.y, f)

model = LinearRegressionModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(0.01).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

for epoch in range(10000):
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})

# Evaluate training accuracy
W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_train, model.y: y_train})
print("W = %s, b = %s, loss = %s" % (W, b, loss))
graph(str(W[0][0])+'*x+'+str(b[0][0]), range(0, 2))

session.close()


