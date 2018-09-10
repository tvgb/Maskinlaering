import sys

import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits import mplot3d
from pylab import meshgrid

y_train, x_train, x2_train = [[0], [1], [1], [1]], [[1], [1], [0], [0]], [[1], [0], [1], [0]]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_train, x2_train, y_train, c='r', marker='o')

ax.set_xlabel('x')
ax.set_ylabel('x2')
ax.set_zlabel('y')


class SigmoidModel:
    def __init__(self):
        # Model input
        self.y = tf.placeholder(tf.float32)
        self.x = tf.placeholder(tf.float32)
        self.x2 = tf.placeholder(tf.float32)

        # Model variables
        self.W1 = tf.Variable([[0.0]])
        self.W2 = tf.Variable([[0.0]])
        self.b1 = tf.Variable([[0.0]])
        self.b2 = tf.Variable([[0.0]])

        # First layer function
        self.h = tf.sigmoid(self.x2 * self.x * self.W1 + self.b1)

        # Predictor
        f = tf.sigmoid(self.h * self.W2 + self.b2)

        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=f)


model = SigmoidModel()


# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(0.1).minimize(model.loss)
iterations = 50000


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
    session.run(minimize_operation, {model.x: x_train, model.x2: x2_train, model.y: y_train})

print('\n\n', 'RESULT: ')

# Evaluate training accuracy
W1, W2, b1, b2, loss = session.run([model.W1, model.W2, model.b1, model.b2, model.loss],
                             {model.x: x_train, model.x2: x2_train, model.y: y_train})
print("W1 = %s, W2 = %s, b1 = %s, b2 =%s, loss = %s" % (W1, W2, b1, b2, loss))

def sigmoid(t):
    return 1 / (1 + np.exp(-t))


x = np.arange(np.min(x_train), np.max(x_train) + 0.1, 0.1)
x2 = np.arange(np.min(x2_train), np.max(x2_train) + 0.1, 0.1)
X, X2 = meshgrid(x, x2)
h = sigmoid((X2*X*W1) + b1)
Y = sigmoid((h*W2) + b2)
ax.plot_wireframe(X, X2, Y)
plt.show()
session.close()
