import os
import pprint as pp
from gensim.models import Word2Vec


def main():
    model = Word2Vec.load(os.path.join('models', 'vectors'))
    test_words = ['food', 'river', 'bank', 'swim']
    for word in test_words:
        print(word)
        pp.pprint(model.similar_by_word(word, topn=10))
    print(model.similarity('bank', 'money'))

    '''
    # Check scores between bank and:
    check
    job
    loan
    funds
    account
    '''


if __name__ == '__main__':
    main()
