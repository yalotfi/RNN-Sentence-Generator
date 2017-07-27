import numpy as np
import csv
import pprint as pp

from keras.models import load_model
from keras.preprocessing.text import Tokenizer


def get_probs(seq, tokenizer, rnn):
    probs = []
    seq = tokenizer.texts_to_sequences([seq])[0]
    seq = np.array(seq)[None]  # reshape as (1, seq_length)
    for idx in range(seq.shape[-1] - 1):
        # get prob of next word
        prob = rnn.predict_on_batch(seq[:, idx])[0, -1][seq[0, idx + 1]]
        probs.append(prob)
    rnn.reset_states()
    # return probability of each word in sequence
    return probs


def preprocess_stories(story_path):
    with open(story_path, 'r') as f:
        train_stories = [story for story in csv.reader(f)]

    # sentences in stories are comma-separated, so join them
    return [' '.join(story) for story in train_stories]


def main(story_path, model_path):
    stories = preprocess_stories(story_path)
    # Tokenize the stories
    tokenizer = Tokenizer(lower=True, filters='')
    # split stories into words, assign number to each unique word
    tokenizer.fit_on_texts(stories)
    pp.pprint(list(tokenizer.word_index.items())[:20])

    print("\nLoading Model...\n")
    rnn = load_model(model_path)
    print("\nModel Ready...\n")

    seq1 = "Jane was working at a diner. Suddenly, a customer barged up to the counter."
    seq2 = "Jane was working at a diner. Dave swam away from the shark."
    probs1 = get_probs(seq1, tokenizer, rnn)
    probs2 = get_probs(seq2, tokenizer, rnn)

    # Format Log Output
    print("")  # Separate from Keras + TF Output
    print("First Sequence of Probabilites:")
    print("{}".format(seq1))
    print("{}\n".format(probs1))
    print("String Len: {} | Mean Prob {}\n".format(
        len(seq1.split(" ")), np.mean(probs1)
    ))
    print("Next Sequence of Probabilites:")
    print("{}".format(seq2))
    print("{}\n".format(probs2))
    print("String Len: {} | Mean Prob {}\n".format(
        len(seq2.split(" ")), np.mean(probs2)
    ))


if __name__ == '__main__':
    main('example_train_stories.csv', 'model/rnn_model.h5')
