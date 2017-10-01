import os
import gensim
import logging

from nltk import sent_tokenize
from nltk import word_tokenize


def main():
    # utiliy functions
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO
    )
    file_path = os.path.join('data', 'roc_grimm_corpus.txt')
    # preprocessing
    corpus = open(file_path).read().lower()
    sentences = [word_tokenize(sent) for sent in sent_tokenize(corpus)]
    windows = [3, 5, 7, 9, 11]
    for w in windows:
        fname = 'vecs_300_window' + str(w)
        model_path = os.path.join('models', 'embeddings', 'roc_grimm', fname)
        model = gensim.models.Word2Vec(
            sentences, size=300, window=w, min_count=5, workers=4)
        model.save(model_path)


if __name__ == '__main__':
    main()
