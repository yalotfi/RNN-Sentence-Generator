import csv
import pprint as pp

from keras.preprocessing.text import Tokenizer


def load_corpus(read):
    with open(read, 'r') as csvfile:
        story_reader = csv.reader(csvfile, delimiter=',')
        story_lists = [row for row in story_reader]
    return [' '.join(story) for story in story_lists]


def main():
    stories = load_corpus('data/test_generation.csv')
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(stories)
    story_vecs = tokenizer.texts_to_sequences(stories)
    word_list = stories[0].split(' ')
    lookup = tokenizer.word_index

    print(len(stories[0]), len(story_vecs[0]), len(word_list[0]))
    pp.pprint(stories[0])
    print("")
    print(story_vecs[0])
    print("")
    vec = []
    for word in word_list:
        word = word.strip('.').strip(',')
        vec.append(lookup.get(word.lower()))
    print(vec)
    print("\n", lookup.get('win,'))


if __name__ == '__main__':
    main()
