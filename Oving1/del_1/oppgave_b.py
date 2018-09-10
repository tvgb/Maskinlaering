import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits import mplot3d
from pylab import meshgrid



data = np.genfromtxt('/Users/trym/Documents/AI/ØVINGER/OVING del 1/del 1/data/day_length_weight.csv', delimiter=',')
y_train, x_train, x2_train = np.hsplit(data, 3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_train, x2_train, y_train, c='r', marker='o')

ax.set_xlabel('Lengde')
ax.set_ylabel('Vekt')
ax.set_zlabel('Dager')


class LinearRegressionModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.x2 = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable([[0.0]])
        self.W2 = tf.Variable([[0.0]])
        self.b = tf.Variable([[0.0]])

        # Predictor
        f = tf.matmul(self.x, self.W) + tf.matmul(self.x2, self.W2) + self.b

        # Uses Mean Squared Error,
        # although instead of mean, sum is used.
        # self.loss = tf.reduce_sum(tf.square(f - self.y))
        self.loss = tf.losses.mean_squared_error(self.y, f)

model = LinearRegressionModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(0.0000006).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

for epoch in range(500000):
    session.run(minimize_operation, {model.x: x_train, model.x2: x2_train, model.y: y_train})

# Evaluate training accuracy
W, W2, b, loss = session.run([model.W, model.W2, model.b, model.loss], {model.x: x_train, model.x2: x2_train, model.y: y_train})
print("W = %s, W2 = %s, b = %s, loss = %s" % (W, W2, b, loss))


x = np.arange(np.min(x_train), np.max(x_train),1)
y = np.arange(np.min(x2_train), np.max(x2_train),1)
X,Y = meshgrid(x, y)
Z = W[0][0]*X + W2[0][0]*Y + b[0][0]
ax.plot_surface(X, Y, Z)
plt.show()

session.close()