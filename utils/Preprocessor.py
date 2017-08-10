import os
import csv
import itertools
import nltk
import numpy as np

# TESTING DEPENDENCIES --------------------------
import random
import sys
import pprint as pp

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import RMSprop
# TESTING DEPENDENCIES --------------------------


# --------------------------------------------
# TESTING FUNCTIONS --------------------------
def build_rnn(vocab_size, embeddings, hh, batch_input_shape):
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
    model.add(Embedding(batch_input_shape=batch_input_shape,
                        input_dim=vocab_size + 1,
                        output_dim=embeddings,
                        mask_zero=True))
    model.add(LSTM(hh,
                   activation='relu',
                   return_sequences=True,
                   stateful=True))
    model.add(Dense(vocab_size + 1))
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
# --------------------------------------------
# TESTING FUNCTIONS --------------------------


def tag_sentences(stories):
    """
    Take a list of stories append start/end tags: return list of list
    """
    # Start and end of sentence tokens
    start_token = 'SENTENCE_START'
    end_token = 'SENTENCE_END'

    # Tokenize stories into sentences
    sentences = [nltk.sent_tokenize(story) for story in stories]

    # Tag each sentence of each story with start + end tags
    tagged_stories = []
    for story in sentences:
        tagged_stories.append(['%s %s %s' % (start_token, sent, end_token)
                               for sent in story])
    return tagged_stories


def load_stories(csv_path):
    print('Reading ROC Stories...\n')
    with open(csv_path, 'r') as csvfile:
        story_reader = csv.reader(csvfile, delimiter=',')
        story_lists = [row for row in story_reader]
        stories = [' '.join(story).lower() for story in story_lists]
        tagged_stories = tag_sentences(stories)
    return tagged_stories


def word_tokenizer(tagged_stories):
    """
    Take a list of stories and tokenize each sentence of each story
    """
    return [[nltk.word_tokenize(sent) for sent in story]
            for story in tagged_stories]


def create_training_data(word2idx, tokenized_sents):
    X_train = np.asarray(
        [[word2idx[w] for w in sent[:-1]] for sent in tokenized_sents])
    y_train = np.asarray(
        [[word2idx[w] for w in sent[1:]] for sent in tokenized_sents])
    return (X_train, y_train)


def preprocess(story_path, vocab_size=8000):
    """
    Load up ROC Stories and return two-way indexes
    """
    tagged_stories = load_stories(story_path)
    tokenized_stories = word_tokenizer(tagged_stories)
    tokenized_sents = list(itertools.chain.from_iterable(tokenized_stories))

    # Count word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sents))
    n_words = len(word_freq.items())
    print('There are {} unique word tokens\n'.format(n_words))

    print('Building word-index dictionaries...')
    # Pull most common words based on defined vocab size
    if n_words > vocab_size:
        vocab = word_freq.most_common(vocab_size - 1)
        print('Vocabulary size is {}'.format(vocab_size))
    else:
        vocab = word_freq.most_common(n_words - 1)
        print('Vocabulary size is {}'.format(n_words))

    # Special token for infrequently used words
    unknown_token = 'UNKNOWN_TOKEN'

    # Build dictionaries: word <-> idx
    idx2word = [word[0] for word in vocab]
    idx2word.append(unknown_token)
    word2idx = dict([(w, i) for i, w in enumerate(idx2word)])
    print('The least frequent word is {}, appearing {} time(s)\n'.format(
        vocab[-1][0], vocab[-1][1])
    )

    # Replace words left out of vocabulary, if any, with "unknown token"
    for i, sent in enumerate(tokenized_sents):
        tokenized_sents[i] = [w if w in word2idx else unknown_token for w in sent]

    # Vectorize tokenized sentences
    print('Vectorizing...\n')
    (X_train, y_train) = create_training_data(word2idx, tokenized_sents)

    # Return two tuples: indexes and training data
    return (word2idx, idx2word), (X_train, y_train)


def main(story_path):
    indexes, train_set = preprocess(story_path)
    X_train, y_train = train_set[0], train_set[1]
    word2idx, idx2word = indexes[0], indexes[1]

    # --------------------------------------------------
    # TESTING ------------------------------------------
    # --------------------------------------------------
    # Model hyperparameters
    max_len = max([len(sequence) for sequence in X_train])
    vocab_size = len(word2idx)
    embeddings = 300
    hidden = 128
    batch_size = 32
    steps = max_len - 1

    # Create (m, n) padded input matrices...
    X_train = pad_sequences(X_train, maxlen=max_len)[:, :-1]
    y_train = pad_sequences(y_train, maxlen=max_len)[:, 1:]
    print(X_train[0: 0 + batch_size].shape)

    # Compile and summarize architecture
    model = build_rnn(vocab_size=vocab_size,
                      embeddings=embeddings,
                      hh=hidden,
                      batch_input_shape=(batch_size, steps))
    print(model.summary())

    # Training
    epochs = 30
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1)
    model.save_weights('rnn_weights.h5')

    # # Generator hyperparameters
    # max_iter = 30
    # sampling_freedom = [0.5, 1.0, 1.2]
    # for iteration in range(1, max_iter + 1):
    #     print()
    #     print("-" * 50)
    #     print("Iteration ", iteration)
    #     model.fit(X_train, y_train,
    #               batch_size=batch_size,
    #               epochs=epochs)
    #     start_index = random.randint(0, len(text) - max_len - 1)
    #     for diversity in sampling_freedom:
    #         print()
    #         print("---- diversity: ", diversity)
    #         generated = ''
    #         sentence = text[start_index: start_index + max_len]
    #         generated += sentence
    #         print('---- Generated with seed: "{}"'.format(sentence))
    #         sys.stdout.write(generated)
    #         for i in range(400):
    #             x = np.zeros((1, max_len, len(chars)))
    #             for t, char in enumerate(sentence):
    #                 x[0, t, word2idx[char]] = 1.
    #             preds = model.predict(x, verbose=0)[0]
    #             next_index = sample(preds, diversity)
    #             next_char = idx2word[next_index]
    #             generated += next_char
    #             sentence = sentence[1:] + next_char
    #             sys.stdout.write(next_char)
    #             sys.stdout.flush()
    #         print()

    # --------------------------------------------------
    # TESTING ------------------------------------------
    # --------------------------------------------------


if __name__ == '__main__':
    story_path = os.path.join('data', 'sample_stories.csv')
    main(story_path)
