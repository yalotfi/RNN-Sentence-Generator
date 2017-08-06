import numpy as np
import random
import sys

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file


def build_rnn(hh, input_shape):
    '''
    --- Compile a Keras RNN model ---
    Very simple LSTM with a single layer of hidden units that are fully
    connected to a softmax layer (crossentropy loss). It takes an input tensor
    of dims: (max sequence length, vocabulary size). Loss is minimized by
    RMSprop which is part of a family of stochastic gradient descent (SGD)
    algorithims that use adaptive learning rates that decay over time.
    '''
    print("Building Model...")
    model = Sequential()
    model.add(LSTM(hh, input_shape=input_shape))
    model.add(Dense(input_shape[1]))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
    return model


def sample(preds, temperature=1.0):
    '''
    --- Helper function to sample from softmax output ---
    Takes input prediction vector and a temperature parameter. Larger temp
    will generate more diverse samples at the cost of accuracy while small
    values will be more confident but conservative.
    '''
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def main():
    # Get training text
    path = get_file(
        'nietzsche.txt',
        origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
    text = open(path).read().lower()
    print('\nCorpus Length: {}\n'.format(len(text)))

    # Create lookup table of character mappings to indexes
    chars = sorted(list(set(text)))
    print('Total Chars: {}\n'.format(len(chars)))
    chars2indices = dict((c, i) for i, c in enumerate(chars))
    indices2chars = dict((i, c) for i, c in enumerate(chars))

    # Split corpus into sequences of input chars (X) and output chars (y)
    maxlen = 40
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print("nb sequences: {}\n".format(len(sentences)))

    # Vectorize the character sequences to numerical tensors
    print("Vectorization...")
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, chars2indices[char]] = 1
        y[i, chars2indices[next_chars[i]]] = 1
    print("X: {}\ny: {}\n".format(X.shape, y.shape))

    # Compile the Keras model
    model = build_rnn(hh=128, input_shape=(maxlen, len(chars)))

    # Bulk of the work starts here...
    for iteration in range(1, 60):
        print()
        print("-" * 50)
        print("Iteration ", iteration)

        # Train model
        model.fit(X, y,
                  batch_size=128,
                  epochs=1)

        # Sample text at 4 different "diversity" settings - see sample() docs
        start_index = random.randint(0, len(text) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print("---- diversity: ", diversity)

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            print('---- Generated with seed: "{}"'.format(sentence))
            sys.stdout.write(generated)

            # Actual generation occurs here:
            for i in range(400):
                x = np.zeros((1, maxlen, len(chars)))

                # Map between output sequence and char dictionary
                for t, char in enumerate(sentence):
                    x[0, t, chars2indices[char]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices2chars[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()
