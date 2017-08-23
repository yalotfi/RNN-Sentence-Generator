import os

from models.architectures import lstm_basic
from utils.Preprocessors import CharProcessor


def main(txt_path):
    # Preprocessing
    maxlen = 20
    stepsize = 1
    corpus = open(txt_path).read().lower()
    char_level = CharProcessor(corpus, maxlen=maxlen, step=stepsize)
    (X_train, y_train) = char_level.training_set_char()

    # Hyperparams
    hidden = 128
    vocab_size = len(char_level.char_list)

    print('Compiling Model...\n')
    model = lstm_basic(hidden, maxlen, vocab_size)
    print(model.summary())


if __name__ == '__main__':
    txt_path = os.path.join('data', 'sample_roc_text.txt')
    main(txt_path)
