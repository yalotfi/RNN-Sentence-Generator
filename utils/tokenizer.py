import os
import csv
import itertools
import nltk
import numpy as np

import pprint as pp


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


def word_tokenizer(tagged_stories):
    """
    Take a list of stories and tokenize each sentence of each story
    """
    return [[nltk.word_tokenize(sent) for sent in story]
            for story in tagged_stories]


def build_indexes(token_sents, vocab_size=8000):
    """
    Take a list of tokenized stories to build vocab indexes
    """
    # Token for infrequently used words
    unknown_token = 'UNKNOWN_TOKEN'

    # Count word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*token_sents))
    n_words = len(word_freq.items())
    print('There are {} unique word tokens\n'.format(n_words))

    # Pull most common words based on defined vocab size
    if n_words > vocab_size:
        vocab = word_freq.most_common(vocab_size - 1)
        print('Vocabulary size is {}\n'.format(vocab_size))
    else:
        vocab = word_freq.most_common(n_words - 1)
        print('Vocabulary size is {}\n'.format(n_words))

    # Build indexes
    idx2word = [word[0] for word in vocab]
    idx2word.append(unknown_token)
    word2idx = dict([(w, i) for i, w in enumerate(idx2word)])
    print('The least frequent word is {}, appearing {} time(s)\n'.format(
        vocab[-1][0], vocab[-1][1])
    )

    # Replace words left out of vocabulary, if any, with "unknown token"
    for i, sent in enumerate(token_sents):
        token_sents[i] = [w if w in word2idx else unknown_token for w in sent]
    return (word2idx, idx2word)


def load_stories(csv_path):
    print('Reading ROC Stories...\n')
    with open(csv_path, 'r') as csvfile:
        story_reader = csv.reader(csvfile, delimiter=',')
        story_lists = [row for row in story_reader]
        stories = [' '.join(story).lower() for story in story_lists]
        tagged_stories = tag_sentences(stories)
    return tagged_stories


def create_training_data(indexes, stories):
    word2idx, idx2word = indexes[0], indexes[1]
    X_train = np.asarray([[word2idx[w] for w in sent[:-1]] for sent in tokens])
    y_train = np.asarray([[word2idx[w] for w in sent[1:]] for sent in tokens])
    return (X_train, y_train)


def main(story_path):
    stories = load_stories(story_path)
    tokenized_stories = word_tokenizer(stories)
    tokenized_sents = list(itertools.chain.from_iterable(tokenized_stories))
    (word2idx, idx2word) = build_indexes(tokenized_sents)
    (X_train, y_train) = create_training_data()
    X_train = np.asarray(
        [[word2idx[w] for w in sent[:-1]] for sent in tokenized_sents])
    y_train = np.asarray(
        [[word2idx[w] for w in sent[1:]] for sent in tokenized_sents])
    print('Train Dim: {}  |  Sample: {}'.format(X_train.shape, X_train[0]))
    print('Label Dim: {}  |  Sample: {}'.format(y_train.shape, y_train[0]))


if __name__ == '__main__':
    story_path = os.path.join('data', 'sample_stories.csv')
    main(story_path)
