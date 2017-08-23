import itertools
import numpy as np

from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import FreqDist


class PreProcessor(object):
    """ Preprocessing over the raw text corpus """

    def __init__(self, corpus, maxlen, step):
        super(PreProcessor, self).__init__()
        self.corpus = corpus
        self.maxlen = maxlen
        self.step = step

    def slice(self, corpus, maxlen, step):
        '''
        Return tuples of input/output sequences by slicing text corpus into
        sequences of length (maxlen), stepping over (step) tokens.
        '''
        seqs_in, seqs_out = [], []
        for i in range(0, len(corpus) - maxlen, step):
            seqs_in.append(corpus[i: i + maxlen])
            seqs_out.append(corpus[i + maxlen])
        return (seqs_in, seqs_out)

    def one_hot_encode(self, token2idx, sequences, vocab_size):
        '''
        Encode input/output sequences into one-hot vectors.
        Returns 3D tensor, X, and 2D tensor, y, of dimensions:
            X.shape = (n_seq, maxlen, vocab_size)
            y.shape = (n_seq, vocab_size)
        '''
        seqs_in, seqs_out = sequences[0], sequences[1]
        X_shape = (len(seqs_in), self.maxlen, vocab_size)
        y_shape = (len(seqs_out), vocab_size)
        # Initialize zero tensors
        X = np.zeros(X_shape, dtype=np.bool)
        y = np.zeros(y_shape, dtype=np.bool)
        # Encode the training examples
        for i, seq in enumerate(seqs_in):
            for t, char in enumerate(seq):
                X[i, t, token2idx[char]] = 1
            y[i, token2idx[seqs_out[i]]] = 1
        return (X, y)


class CharProcessor(PreProcessor):
    """
    Preprocesses corpus into char-level indexes and training examples
    """

    def __init__(self, corpus, maxlen, step):
        super(CharProcessor, self).__init__(corpus, maxlen, step)
        self.char_list = self._char_list(corpus)

    def _char_list(self, corpus):
        '''
        Return a sorted list of unique characters in the text corpus
        '''
        return sorted(list(set(corpus)))

    def get_char2idx(self, char_list):
        '''
        Create a char to index dictionary to create training set
        '''
        return dict((c, i) for i, c in enumerate(char_list))

    def get_idx2char(self, char_list):
        '''
        Create index to char dictionary when generating new text
        '''
        return dict((i, c) for i, c in enumerate(char_list))

    def training_set_char(self):
        '''
        Return train set (X, y) binary class vectors for char level model
        '''
        char2idx = self.get_char2idx(self.char_list)
        sequences = self.slice(self.corpus, self.maxlen, self.step)
        vocab_size = len(self.char_list)
        return self.one_hot_encode(char2idx, sequences, vocab_size)


class WordProcessor(PreProcessor):
    """docstring for WordProcessor"""

    def __init__(self, corpus, maxlen, step, vocab_size=8000):
        super(WordProcessor, self).__init__(corpus, maxlen, step)
        self.vocab_size = vocab_size
        self.token_vecs = self._tag_sentences(corpus)

    def _tag_sentences(self, corpus):
        '''
        Return a list of tagged and tokenized sentences
        '''
        # Start and end of sentence tokens
        start_token = 'SENTENCE_START'
        end_token = 'SENTENCE_END'
        # Tokenize stories into sentences
        sentences = sent_tokenize(corpus)
        # Tag each sentence of each story with start + end tags
        token_vecs = []
        for sent in sentences:
            token_vecs.append('%s %s %s' % (start_token, sent, end_token))
        # Return list of word tokenized sentences
        return [word_tokenize(sent) for sent in token_vecs]

    def _build_vocab(self, token_vecs):
        # Count word frequencies
        word_freq = FreqDist(itertools.chain(*token_vecs))
        n_words = len(word_freq.items())
        # Pull most common words based on defined vocab size
        if n_words > self.vocab_size:
            return word_freq.most_common(self.vocab_size - 1)
        else:
            self.vocab_size = n_words
            return word_freq.most_common(n_words - 1)

    def _map_indexes(self, token_vecs):
        # Special token for infrequently used words and sequence pads
        unknown_token = 'UNKNOWN_TOKEN'
        # Create vocabulary
        vocab = self._build_vocab(token_vecs)
        # Build dictionaries: word <-> idx
        idx2word = [word[0] for word in vocab]
        idx2word.append(unknown_token)
        word2idx = dict([(w, i) for i, w in enumerate(idx2word)])
        # Replace words left out of vocabulary, if any, with "unknown token"
        for i, sent in enumerate(token_vecs):
            token_vecs[i] = [
                w if w in word2idx else unknown_token for w in sent]
        return (word2idx, idx2word)

    def get_word2idx(self):
        return self._map_indexes(self.token_vecs)[0]

    def get_idx2word(self):
        return self._map_indexes(self.token_vecs)[1]

    def training_set_word(self):
        '''
        Return train set (X, y) binary class vectors for word level model
        '''
        word2idx = self.get_word2idx()
        word_corpus = list(itertools.chain(*self.token_vecs))
        sequences = self.slice(word_corpus, self.maxlen, self.step)
        return self.one_hot_encode(word2idx, sequences, self.vocab_size)
