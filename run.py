import os

from models.architectures import lstm_basic
from preprocessing.Preprocessor import Preprocessor


def main(txt_path):
    print('Preprocessing..\n')
    corpus = open(txt_path).read().lower()
    PreProc = Preprocessor(corpus, maxlen=40, step=1)

    print('Corpus Length: {}\nVocab Size: {}\n'.format(
        len(PreProc.corpus), len(PreProc.vocab))
    )
    print('Dictionary Map: {} | {}\n'.format(
        PreProc.char2idx['.'], PreProc.idx2char[6]))
    print('Number of sequences: {}\n'.format(
        PreProc.X.shape[0])
    )
    print('X: {}\ny: {}\n'.format(PreProc.X.shape, PreProc.y.shape))

    print('Compiling Model...\n')
    hidden = 128
    max_len = 20
    vocab_size = 100

    model = lstm_basic(hidden, max_len, vocab_size)
    print(model.summary())


if __name__ == '__main__':
    txt_path = os.path.join('data', 'sample_roc_text.txt')
    main(txt_path)
