"""
Implementation of RNN for fitting a language model to the Penn Treebank dataset.

Adapted from: http://www.tensorflow.org/tutorials/recurrent/index.html#lstm
"""

from os import path
import numpy as np
import time

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell, seq2seq
from keras.preprocessing.text import Tokenizer

__author__ = 'sebastian'


def read_file(file_path):
    """
    Read file, replace newlines with end-of-sentence marker.
    :param filename:
    :return the file as a string
    """

    with open(file_path, 'r') as f:
        return f.read().replace('\n', '<eos>')


def ptb_iterator(raw_data, batch_size, num_steps):
    """
    Generates generates batch_size pointers into the raw PTB data, and allows
    minibatch iteration along these pointers.
    :param raw_data: one of the raw data outputs from ptb_raw_data.
    :param batch_size: int, the batch size.
    :param num_steps: int, the number of unrolls.
    :return Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
        The second element of the tuple is the same data time-shifted to the
        right by one.
    :raises ValueError: if batch_size or num_steps are too high.
    """
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len / batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) / num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)

if __name__ == '__main__':

    # parameters
    batch_size = 20  # the batch size
    num_steps = 20  # the number of unrolled steps of LSTM
    vocab_size = 10000  # the size of the vocabulary
    hidden_size = 200  # the number of LSTM units
    num_layers = 1  # the number of LSTM layers
    max_grad_norm = 5  # maximum number at which gradient should be clipped
    lr = 1.0  # learning rate
    max_epoch = 20  # the maximum number of epochs
    lr_decay = 0.5  # decay of the learning rate

    ptb_path = 'data/simple-examples/data/'
    file_paths = [path.join(ptb_path, 'ptb.%s.txt' % part) for part in ['train', 'valid', 'test']]

    # read in the corpora; note: we only use train_corpus at the moment for test purposes
    train_corpus, valid_corpus, test_corpus = [read_file(file_path) for file_path in file_paths]

    # convert words to integers; for convenience, we use Keras' tokenizer class
    print 'Training the vectorizer.'
    tokenizer = Tokenizer(nb_words=vocab_size)
    tokenizer.fit_on_texts([train_corpus])

    print 'Transforming sentences into index sequences.'
    train_data = tokenizer.texts_to_sequences([train_corpus])[0]

    print 'Building the model'

    # define placeholders for input and output values
    input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    targets = tf.placeholder(tf.int32, [batch_size, num_steps])

    # create the LSTM cell
    lstm_cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0)

    # add dropout here if needed

    # stack LSTM cells
    cell = rnn_cell.MultiRNNCell([lstm_cell] * num_layers)

    # create the initial state of the cell
    initial_state = cell.zero_state(batch_size, tf.float32)

    # get embeddings
    embedding = tf.get_variable('embedding', [vocab_size, hidden_size])
    inputs = tf.split(
        1, num_steps, tf.nn.embedding_lookup(embedding, input_data))
    inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

    # add dropout here if needed

    # create outputs and states
    outputs, states = rnn.rnn(cell, inputs, initial_state=initial_state)

    # reshape output
    output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])

    # specify XW + b
    logits = tf.nn.xw_plus_b(output,
                             tf.get_variable('softmax_w', [hidden_size, vocab_size]),
                             tf.get_variable('softmax_b', [vocab_size]))

    # define loss
    loss = seq2seq.sequence_loss_by_example([logits],
                                            [tf.reshape(targets, [-1])],
                                            [tf.ones([batch_size * num_steps])],
                                            vocab_size)

    # define individual cost
    _cost = tf.reduce_sum(loss) / batch_size

    # get final state
    final_state = states[-1]

    # create learning rate variable
    _lr = tf.Variable(0.0, trainable=False)

    # define gradient clipping
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(_cost, tvars), max_grad_norm)

    # create optimizer and define application of gradients (2nd part of minimize)
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    # start the session
    with tf.Session() as session:
        tf.initialize_all_variables().run()

        # iterate over the epochs
        for i in range(max_epoch):
            lr_decay = lr_decay ** max(i - max_epoch, 0.0)
            session.run(tf.assign(_lr, lr * lr_decay))

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(_lr)))

            # specify parameters for mini-batch
            epoch_size = ((len(train_data) / batch_size) - 1) / num_steps
            start_time = time.time()
            costs = 0.0
            iters = 0
            state = initial_state.eval()
            verbose = True

            # iterate in mini-batches over the dataset
            for step, (x, y) in enumerate(ptb_iterator(train_data, batch_size, num_steps)):
                cost, state, _ = session.run([_cost, final_state, train_op],
                                             {input_data: x, targets: y, initial_state: state})
                costs += cost
                iters += num_steps

                if verbose and step % (epoch_size / 10) == 10:
                    print("%.3f perplexity: %.3f speed: %.0f wps" %
                        (step * 1.0 / epoch_size, np.exp(costs / iters),
                        iters * batch_size / (time.time() - start_time)))

            # calculate perplexity
            train_perplexity = np.exp(costs / iters)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
