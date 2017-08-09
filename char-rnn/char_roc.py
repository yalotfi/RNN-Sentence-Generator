import os
import sys
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop


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


def main(txt_path):
    # read corpus and create vocabulary of language model
    corpus = open(txt_path).read().lower()
    vocab = sorted(list(set(corpus)))

    # dictionary of char indexes
    chars2idxs = dict((c, i) for i, c in enumerate(vocab))
    idxs2chars = dict((i, c) for i, c in enumerate(vocab))

    # split corpus into input/output sequences for rnn
    maxlen = 40
    step = 3
    sentences, next_chars = [], []
    for i in range(0, len(corpus) - maxlen, step):
        sentences.append(corpus[i: i + maxlen])
        next_chars.append(corpus[i + maxlen])

    # vectorize the sequences using the [char: idx]
    print('Vectorization...\n')
    X = np.zeros((len(sentences), maxlen, len(vocab)), dtype=np.bool)
    y = np.zeros((len(sentences), len(vocab)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, chars2idxs[char]] = 1
        y[i, chars2idxs[next_chars[i]]] = 1

    # Compile the Keras model
    print("Building Model...")
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(vocab))))
    model.add(Dense(len(vocab)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))

    # console log
    print('Corpus Length: {}\nVocab Size: {}\n'.format(
        len(corpus), len(vocab)))
    print('Dictionary Map: {} | {}\n'.format(
        chars2idxs['.'], idxs2chars[6]))
    print('Number of sequences: {}\n'.format(len(sentences)))
    print('X: {}\ny: {}\n'.format(X.shape, y.shape))
    print(model.summary())

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
        start_index = random.randint(0, len(corpus) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print("---- diversity: ", diversity)

            generated = ''
            sentence = corpus[start_index: start_index + maxlen]
            generated += sentence
            print('---- Generated with seed: "{}"'.format(sentence))
            sys.stdout.write(generated)

            # Actual generation occurs here:
            for i in range(400):
                x = np.zeros((1, maxlen, len(vocab)))

                # Map between output sequence and char dictionary
                for t, char in enumerate(sentence):
                    x[0, t, chars2idxs[char]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = idxs2chars[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()


if __name__ == '__main__':
    txt_path = os.path.join('data', 'roc_text.txt')
    print('\nReading text from {}\n'.format(txt_path))
    main(txt_path)
