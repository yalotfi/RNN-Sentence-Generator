import os
import numpy as np


class Preprocessor():
    """
    Preprocesses data into model inputs and useful tools like indexes

    Initializing the Preprocessor will execute the necessary methods,
    thereby storing the following:
        1. Text corpus
        2. Max sequence length
        3. Step size
        4. Vocab size - n labels
        5. Set of token-int dictionaries
        6. Training examples (X, y)
    """

    def __init__(self, corpus, maxlen, step):
        super(Preprocessor, self).__init__()
        self.corpus = corpus
        self.maxlen = maxlen
        self.step = step
        self.vocab = sorted(list(set(self.corpus)))
        (self.char2idx, self.idx2char) = self._index_char(self.vocab)
        (self.X, self.y) = self._vectorize()

    def _index_char(self, vocab):
        '''
        Create two dictionaries that index every token to an integer value.
        Used for both training data creation as well as sequence generation.
        '''
        char2idx = dict((c, i) for i, c in enumerate(vocab))
        idx2char = dict((i, c) for i, c in enumerate(vocab))
        return (char2idx, idx2char)

    def _sequence_io(self, corpus, maxlen, step):
        '''
        Split the text corpus into sequences of (maxlen) tokens. Output label
        sequence is offset one token ahead of input sequence.
        '''
        seqs_in, seqs_out = [], []
        for i in range(0, len(corpus) - maxlen, step):
            seqs_in.append(corpus[i: i + maxlen])
            seqs_out.append(corpus[i + maxlen])
        return (seqs_in, seqs_out)

    def _vectorize(self):
        '''
        Encode input/output sequences into one-hot vectors.
        Returns 3D tensor, X, with dims (n_seq, max_len, vocab_size)
        and 2D matrix, y, with dims (n_dims, vocab_size)
        '''
        (seqs_in, seqs_out) = self._sequence_io(self.corpus,
                                                self.maxlen,
                                                self.step)
        X = np.zeros((len(seqs_in), self.maxlen, len(self.vocab)),
                     dtype=np.bool)
        y = np.zeros((len(seqs_in), len(self.vocab)),
                     dtype=bool)
        for i, seq in enumerate(seqs_in):
            for t, char in enumerate(seq):
                X[i, t, self.char2idx[char]] = 1
            y[i, self.char2idx[seqs_out[i]]] = 1
        return (X, y)
