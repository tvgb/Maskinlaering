import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


data = np.genfromtxt('/Users/trym/Documents/AI/ØVINGER/OVING del 1/del 1/data/day_head_circumference.csv', delimiter=',')
x_train, y_train = np.hsplit(data, 2)

plt.plot(x_train, y_train, 'ro')
#plt.show()


def graph(formula, x_range):
    x = np.array(x_range)
    y = eval(formula)
    plt.plot(x, y)
    plt.xlabel('Dager')
    plt.ylabel('Hodeomkrets')
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
        f = (20*tf.sigmoid(tf.matmul(self.x, self.W) + self.b)) + 31

        # Uses Mean Squared Error,
        # although instead of mean, sum is used.
        # self.loss = tf.reduce_sum(tf.square(f - self.y))
        self.loss = tf.losses.mean_squared_error(self.y, f)



model = LinearRegressionModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(0.0000000001).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

for epoch in range(10000):
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})

# Evaluate training accuracy
W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_train, model.y: y_train})
print("W = %s, b = %s, loss = %s" % (W, b, loss))


graph('20*(del 1/(del 1+2.71828182846**(-(x*'+str(W[0][0])+'+'+str(b[0][0])+')))) + 31', range(0, 2000))

session.close()

