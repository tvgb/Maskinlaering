import sys

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random as r
from mpl_toolkits import mplot3d
from pylab import meshgrid


y_graph, x_graph, x2_graph = [[0], [1], [1], [0]], [[1], [1], [0], [0]], [[1], [0], [1], [0]]

x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_graph, x2_graph, y_graph, c='r', marker='o')

ax.set_xlabel('x')
ax.set_ylabel('x2')
ax.set_zlabel('y')

def get_r():
    return r.uniform(-1, 1)

class SigmoidModel:
    def __init__(self):
        # Model input
        self.y = tf.placeholder(tf.float32)
        self.x = tf.placeholder(tf.float32)


        # Model variables
        self.W1 = tf.Variable([[get_r(), get_r()], [get_r(), get_r()]])
        self.W2 = tf.Variable([[get_r()], [get_r()]])

        self.b1 = tf.Variable([[0.0, 0.0]])
        self.b2 = tf.Variable([[0.0]])

        # First layer function
        self.xwb = tf.matmul(self.x, self.W1) + self.b1
        self.h = tf.sigmoid(self.xwb)

        # Logits
        logits = tf.matmul(self.h, self.W2) + self.b2

        # Loss
        self.loss = tf.losses.sigmoid_cross_entropy(self.y, logits)


model = SigmoidModel()


# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(1).minimize(model.loss)
iterations = 10000


# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

print('Processing...')
for epoch in range(iterations):
    frac = epoch / iterations
    filled_progbar = round(frac*20)
    print('\r','#'*filled_progbar + '-'*(20-filled_progbar), '[{:>7.2%}]'.format(frac), end='')
    sys.stdout.flush()
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})

print('\n\n', 'RESULT: ')

# Evaluate training accuracy
W1, W2, b1, b2, loss = session.run([model.W1, model.W2, model.b1, model.b2, model.loss],
                             {model.x: x_train, model.y: y_train})
print("W1 = %s, W2 = %s, b1 = %s, b2 =%s, loss = %s" % (W1, W2, b1, b2, loss))


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


class XORSigmoigPlot:
    def __init__(self, W1, b1, W2, b2):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

    # Predictor, first layer
    def f1(self, x):
        return sigmoid(x @ self.W1 + self.b1)  # HUSK ALLTID: '@' er matrisemultiplikasjon, ikke '*'!

    # Predictor, second layer
    def f2(self, h):
        return sigmoid(h @ self.W2 + self.b2)

    def f(self, x):
        return self.f2(self.f1(x))

    def loss(self, x, y):
        return -np.mean(np.multiply(y, np.log(self.f(x))) + np.multiply((1 - y), np.log(1 - self.f(x))))


grid_spacing = 25
x1_grid, x2_grid = np.meshgrid(np.linspace(0, np.max(x_train[:, 0]), grid_spacing), np.linspace(0, np.max(x_train[:, 1]), grid_spacing))
y_grid = np.empty([grid_spacing, grid_spacing])

model = XORSigmoigPlot(W1, b1, W2, b2)

for i in range(0, x1_grid.shape[0]):
    for j in range(0, x1_grid.shape[1]):
        x = np.array([[x1_grid[i, j], x2_grid[i, j]]])
        y_grid[i, j] = model.f(x)

ax.plot_wireframe(x1_grid, x2_grid, y_grid, color='blue')

plt.show()

session.close()
