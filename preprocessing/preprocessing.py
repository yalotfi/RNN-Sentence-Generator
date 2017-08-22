import os
import csv
import itertools
import nltk
import numpy as np

# TESTING DEPENDENCIES --------------------------
# import random
# import sys
# import pprint as pp

from models.architecture import lstm_basic

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# from keras.models import Sequential
# from keras.layers.embeddings import Embedding
# from keras.layers import LSTM
# from keras.layers import Dense
# from keras.layers import Activation
# from keras.optimizers import RMSprop
# TESTING DEPENDENCIES --------------------------


# --------------------------------------------
# TESTING FUNCTIONS --------------------------
# --------------------------------------------
# def lstm_basic(hidden, max_len, vocab_size):
#     model = Sequential()
#     model.add(LSTM(hidden, input_shape=(max_len, vocab_size)))
#     model.add(Dense(vocab_size, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
#     return  model


# def lstm_embedding(batch_size, timesteps, vocab_size, embeddings, hidden):
#     print("Building Model...")
#     model = Sequential()
#     model.add(Embedding(batch_input_shape=(batch_size, timesteps),
#                         input_dim=vocab_size + 1,
#                         output_dim=embeddings,
#                         mask_zero=True))
#     model.add(LSTM(hidden, return_sequences=True, stateful=True))
#     model.add(Dense(vocab_size + 1, activation='softmax'))
#     model.compile(loss='sparse_categorical_crossentropy',
#                   optimizer=RMSprop(lr=0.01))
#     return model


# def sample(preds, temperature=1.0):
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     return np.argmax(probas)
# --------------------------------------------
# TESTING FUNCTIONS --------------------------
# --------------------------------------------


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


def build_indexes(tokenized_sents, vocab_size):
    # Count word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sents))
    n_words = len(word_freq.items())
    print('There are {} unique word tokens\n'.format(n_words))

    print('Building word-index dictionaries...')
    # Pull most common words based on defined vocab size
    if n_words > vocab_size:
        vocab = word_freq.most_common(vocab_size - 1)
        print('\tVocabulary size is {}'.format(vocab_size))
    else:
        vocab = word_freq.most_common(n_words - 1)
        print('\tVocabulary size is {}'.format(n_words))

    # Special token for infrequently used words and sequence pads
    unknown_token = 'UNKNOWN_TOKEN'
    padding_token = 'PADDING'

    # Build dictionaries: word <-> idx
    idx2word = [word[0] for word in vocab]
    idx2word.append(unknown_token)
    idx2word.append(padding_token)
    word2idx = dict([(w, i) for i, w in enumerate(idx2word)])
    print('\tThe least frequent word is {}, appearing {} time(s)\n'.format(
        vocab[-1][0], vocab[-1][1])
    )

    # Replace words left out of vocabulary, if any, with "unknown token"
    for i, sent in enumerate(tokenized_sents):
        tokenized_sents[i] = [
            w if w in word2idx else unknown_token for w in sent]

    # Return the two dicts
    return word2idx, idx2word


def create_training_data(word2idx, tokenized_sents):
    # Vectorize tokens and pad to max sequence length
    max_seq = max([len(sent) for sent in tokenized_sents])
    indexed = [[word2idx[w] for w in sent] for sent in tokenized_sents]
    pad_vecs = pad_sequences(indexed, max_seq, value=len(word2idx))

    # Create feature and label sets
    X_train = np.asarray([[w for w in sent[:-1]] for sent in pad_vecs])
    y_train = np.asarray([[w for w in sent[1:]] for sent in pad_vecs])

    # Reshape to 3D tensor
    tensor_dim = (len(pad_vecs), max_seq - 1, 1)
    return (X_train.reshape(tensor_dim), y_train.reshape(tensor_dim))


def preprocess(story_path, vocab_size=8000):
    """
    Process ROC Stories from tokenization through vectorization
    """
    # Load raw text, tokenize sentences and words, then index
    print('Parsing stories...\n')
    tagged_stories = load_stories(story_path)
    tokenized_stories = word_tokenizer(tagged_stories)
    tokenized_sents = list(itertools.chain.from_iterable(tokenized_stories))
    (word2idx, idx2word) = build_indexes(tokenized_sents, vocab_size)

    # Vectorize tokenized sentences into training data
    print('Vectorizing Text Sequences...')
    (X_train, y_train) = create_training_data(word2idx, tokenized_sents)
    print('\tFeature Size: {}  |  Label size {}\n'.format(
        X_train.shape, y_train.shape))

    # Return two tuples: vocab dictionaries and training data
    return (word2idx, idx2word), (X_train, y_train)


def main(story_path):
    (word2idx, idx2word), (X_train, y_train) = preprocess(story_path)

    # --------------------------------------------------
    # TESTING ------------------------------------------
    # --------------------------------------------------
    # Hyperparameters
    batch_size = 64
    embeddings = 64
    max_len = X_train.shape[1]
    steps = max_len - 1
    vocab_size = len(word2idx)
    hidden = 128

    # Compile and summarize architecture
    # model = lstm_embedding(batch_size, steps, vocab_size, embeddings, hidden)
    model = lstm_basic(hidden, max_len, vocab_size)
    print(model.summary())

    # Training
    # epochs = 30
    # model.fit(X_train, y_train,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           verbose=1)
    # model.save_weights('rnn_weights.h5')

    # # Generate
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
