import os
import pprint as pp
import time as t

from gensim.models import Word2Vec


def main():
    print('Looking at these word vectors:')
    test_words = ['money', 'job', 'account', 'fund', 'bank']
    print(test_words)
    windows = [3, 5, 7, 9, 11]
    tic = t.time()
    for w in windows:
        print('\nLOADING MODEL WITH WINDOW SIZE {}\n'.format(w))
        fname = 'vecs_300_window' + str(w)
        model = Word2Vec.load(os.path.join('models', 'embeddings', fname))
        for word in test_words:
            print(word)
            pp.pprint(model.similar_by_word(word, topn=10))
    toc = t.time() - tic
    print("Runtime: {}".format(toc))
    # print(model.similarity('bank', 'money'))


if __name__ == '__main__':
    main()
