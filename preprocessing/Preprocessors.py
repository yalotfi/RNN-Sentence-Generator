import os
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

    def get_char_list(self, corpus):
        '''
        Return a sorted list of unique characters in the text corpus
        '''
        return sorted(list(set(corpus)))

    def slice_corpus(self, corpus, maxlen, step):
        '''
        Return tuples of input/output vectors by slicing text corpus into
        sequences of length (maxlen), stepping over (step) tokens.
        '''
        seqs_in, seqs_out = [], []
        for i in range(0, len(corpus) - maxlen, step):
            seqs_in.append(corpus[i: i + maxlen])
            seqs_out.append(corpus[i + maxlen])
        return (seqs_in, seqs_out)

    def tag_sentences(self, corpus):
        '''
        Return a list of sentences each tagged with start and end tokens
        '''
        # Start and end of sentence tokens
        start_token = 'SENTENCE_START'
        end_token = 'SENTENCE_END'

        # Tokenize stories into sentences
        sentences = sent_tokenize(corpus)

        # Tag each sentence of each story with start + end tags
        tagged_sents = []
        for sent in sentences:
            tagged_sents.append('%s %s %s' % (start_token, sent, end_token))
        return tagged_sents


class CharProcessor(PreProcessor):
    """
    Preprocesses corpus into char-level indexes and training examples
    """

    def __init__(self, corpus, maxlen, step):
        super(CharProcessor, self).__init__(corpus, maxlen, step)
        self.char_list = self.get_char_list(corpus)

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

    def vectorize_char(self):
        '''
        Encode input/output sequences into one-hot vectors.
        Returns 3D tensor, X, and 2D tensor, y, of dimensions:
            X.shape = (n_seq, maxlen, vocab_size)
            y.shape = (n_seq, vocab_size)
        '''
        # Load character to index dictionary
        char2idx = self.get_char2idx(self.char_list)
        # Slice corpus into input/output sequences
        examples = self.slice_corpus(self.corpus, self.maxlen, self.step)
        inputs, outputs = examples[0], examples[1]
        # Define dimensions of training features, X, and labels, y
        X_shape = (len(inputs), self.maxlen, len(self.char_list))
        y_shape = (len(outputs), len(self.char_list))
        # Initialize zero tensors
        X = np.zeros(X_shape, dtype=np.bool)
        y = np.zeros(y_shape, dtype=np.bool)
        # Encode the training examples
        for i, seq in enumerate(inputs):
            for t, char in enumerate(seq):
                X[i, t, char2idx[char]] = 1
            y[i, char2idx[outputs[i]]] = 1
        return (X, y)


class WordProcessor(PreProcessor):
    """docstring for WordProcessor"""

    def __init__(self, corpus, maxlen, step, vocab_size=8000):
        super(WordProcessor, self).__init__(corpus, maxlen, step)
        self.vocab_size = vocab_size
        self.tagged_sents = self.tag_sentences(corpus)
        self.token_sents = self.word_tokenizer(self.tagged_sents)

    def word_tokenizer(self, tagged_sents):
        '''
        Tokenize text corpus at word-level
        '''
        return [word_tokenize(sent) for sent in tagged_sents]


if __name__ == '__main__':
    # Load in text corpus
    txt_path = os.path.join('data', 'sample_roc_text.txt')
    corpus = open(txt_path).read().lower()

    # Data hyperparams
    maxlen = 40
    stepsize = 1

    # Init each type of preprocessor:
    char_level = CharProcessor(corpus, maxlen=maxlen, step=stepsize)
    word_level = WordProcessor(corpus, maxlen=maxlen, step=stepsize)

    # Create character level training set
    (X_char, y_char) = char_level.vectorize_char()

    # Check output
    print('Char Training Data:')
    print(X_char.shape)
    print(y_char.shape)
    print('\nWord Training Data:')
    print(len(word_level.token_sents))
    print(word_level.token_sents[0])
