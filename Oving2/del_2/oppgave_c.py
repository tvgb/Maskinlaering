import numpy as np
import tensorflow as tf
from Oving2.del_2 import encoder as e


class LongShortTermMemoryModel:
    def __init__(self, encodings_size_in, encodings_size_out):
        # Model constants
        cell_state_size = 128

        # Cells
        cell = tf.contrib.rnn.BasicLSTMCell(cell_state_size)

        # Model input
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')  # Needed by cell.zero_state call, and can be dependent on usage (training or generation)
        self.x = tf.placeholder(tf.float32, [None, None, encodings_size_in], name='x')  # Shape: [batch_size, max_time, encodings_size]
        self.y = tf.placeholder(tf.float32, [None, encodings_size_out], name='y')  # Shape: [batch_size, encodings_size]
        self.in_state = cell.zero_state(self.batch_size, tf.float32)  # Can be used as either an input or a way to get the zero state

        # Model variables
        W = tf.Variable(tf.random_normal([cell_state_size, encodings_size_out]))
        b = tf.Variable(tf.random_normal([encodings_size_out]))

        # Model operations
        lstm, self.out_state = tf.nn.dynamic_rnn(cell, self.x, initial_state=self.in_state)  # lstm has shape: [batch_size, max_time, cell_state_size]

        # Logits, where tf.einsum multiplies a batch of txs matrices (lstm) with W
        logits = tf.nn.bias_add(tf.matmul(lstm[:, -1, :], W), b)  # b: batch, t: time, s: state, e: encoding

        # Predictor
        self.f = tf.nn.softmax(logits)

        # Cross Entropy loss
        self.loss = tf.losses.softmax_cross_entropy(self.y, logits)


def train_model(word_list, emoji_list, start_word, epochs, print_every, print_loss, print_y):

    char_encodings, index_to_char = e.get_char_encoder(e.get_alphabet(word_list))  # for cat, rat and matt
    emoji_encoding, index_to_emoji = e.get_emoji_encodings(emoji_list)

    x_train = e.get_x_train(word_list, char_encodings, index_to_char)
    y_train = emoji_encoding

    model = LongShortTermMemoryModel(np.shape(char_encodings)[1], np.shape(emoji_encoding)[1])

    # Training: adjust the model so that its loss is minimized
    minimize_operation = tf.train.RMSPropOptimizer(0.05).minimize(model.loss)

    # Create session object for running TensorFlow operations
    session = tf.Session()

    # Initialize tf.Variable objects
    session.run(tf.global_variables_initializer())

    batch_size = len(word_list)

    # Initialize model.in_state
    zero_state = session.run(model.in_state, {model.batch_size: batch_size})

    for epoch in range(epochs):
        session.run(minimize_operation, {model.batch_size: batch_size, model.x: x_train, model.y: y_train, model.in_state: zero_state})

        if (epoch + 1) % print_every == 0:

            print_string = '\n'

            print_string += 'epoch ' + str(epoch + 1) + ' | '
            if print_loss:
                print_string += 'loss ' + str(session.run(model.loss, {model.batch_size: 1, model.x: x_train, model.y: y_train, model.in_state: zero_state})) + ' | '

            state = session.run(model.in_state, {model.batch_size: 1})
            y, state = session.run([model.f, model.out_state], {model.batch_size: 1, model.x: e.get_x_train([start_word], char_encodings, index_to_char), model.in_state: state})
            print_string += index_to_emoji[y.argmax()]

            if print_y:
                print_string += ' | y ' + str(y)

            print(print_string)

    session.close()


if __name__ == '__main__':
    epochs = 500
    print_every = 50  # Antall epoker mellom hver print

    word_list = ['hat ', 'rat ', 'cat ', 'flat', 'matt', 'cow ', 'sun ']
    emoji_list = [':top_hat:', ':rat:', ':cat:', ':house:', ':man:', ':cow:', ':sun:']

    start_word = 'c'  # Ordet modellen prøver å finne en passende emoji til

    train_model(word_list, emoji_list, start_word, epochs, print_every, print_loss=True, print_y=True)
