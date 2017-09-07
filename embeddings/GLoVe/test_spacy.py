# import numpy as np
import en_core_web_sm
# from spacy.en import English
from spacy.attrs import ORTH


def read_file(file):
    with open(file, 'r') as f:
        return f.read()


def main():
    # init doc object
    # text = read_file('sample_roc_text.txt')
    text = 'David noticed he had put on a lot of weight recently. He examined his habits to try and figure out the reason. He realized he\'d been eating too much fast food lately. He stopped going to burger places and started a vegetarian diet. After a few weeks, he started to feel much better.'
    nlp = en_core_web_sm.load()
    doc = nlp(text)

    # build dictionary
    counts = doc.count_by(ORTH)
    word_ids = list(counts.keys())
    print(len(word_ids))
    # test_word = nlp.vocab.strings[word_ids[0]]

    # play around with GLoVe embeddings
    weight = doc[9]  # weight
    food = doc[33]  # food
    vegetarian = doc[45]  # vegetarian
    diet = doc[46]  # diet
    tokens = [weight, food, vegetarian, diet]

    # word similarity scores
    print('\nsemantic similarity between four words:')
    print(food.similarity(weight))
    print(food.similarity(food))
    print(food.similarity(vegetarian))
    print(food.similarity(diet))
    print(diet.similarity(vegetarian))
    # print('\nVector shape of word, weight')
    # print(weight.vector.shape)
    # print(weight.vector)


    # sentence similarity scores
    sentences = [s for s in doc.sents]
    print('\nsemantic similarity across five sentences')
    print(sentences[0].similarity(sentences[1]))
    print(sentences[1].similarity(sentences[2]))
    print(sentences[2].similarity(sentences[3]))
    print(sentences[3].similarity(sentences[4]))

    # resize vectors
    print('\nprint embeddings of size (64, 1)')
    for token in tokens:
        token.vocab.resize_vectors(64)
        # print(token.vector)

    # # adjusted similarity scores
    # print('\nword score should be different after resizing')
    # print(food.similarity(weight))
    # print(food.similarity(food))
    # print(food.similarity(vegetarian))
    # print(food.similarity(diet))

    # convert to numpy array by attribute
    print('\nwe can also store doc objects as numpy arrays')
    np_array = doc.to_array([ORTH])
    print(np_array.shape)


if __name__ == '__main__':
    main()
