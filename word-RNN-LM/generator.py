import csv
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.models import load_model


def load_corpus(read):
    with open(read, 'r') as csvfile:
        story_reader = csv.reader(csvfile, delimiter=',')
        story_lists = [row for row in story_reader]
    return [' '.join(story) for story in story_lists]


def main(model_path, vocab_path, test_path, gen_iters):
    # Load trained RNN model
    print("\nLoading Model...\n")
    rnn_lm = load_model(model_path)  # First main() param
    print("\nModel Ready...\n")

    # Load and process ROC corpus
    corpus = load_corpus(vocab_path)  # Second main() param

    # Use Keras primitive to tokenize ROC corpus
    token = Tokenizer(lower=True, filters='')
    token.fit_on_texts(corpus)
    print("\nVocab size: {}\n".format(len(token.word_index)))

    # Create a lookup table for the vocab indexes
    vocab_lookup = {index: word for word, index in token.word_index.items()}
    end_of_sentence = ['.', '?', '!']  # eos tokens

    # Load 25 stories to test generative story endings
    with open(test_path, 'r') as csvfile:  # Third main() param
        reader = csv.reader(csvfile)
        test_stories = []
        row = 0
        while row < gen_iters:
            for story in reader:
                # Test file has a blank column, just remove it if it exists
                if len(story) == 6:
                    del story[-1]
                    test_stories.append(story)
                else:
                    test_stories.append(story)
            row += 1
    print("Will generate {} endings.\n".format(len(test_stories)))

    # Preprocess the test stories into their contexts, endings, and sequences
    contexts, endings = [], []
    for story in test_stories:
        contexts.append(' '.join(story[:-1]))
        endings.append(story[-1])
    sequences = token.texts_to_sequences(contexts)

    # Generate endings for the first i stories loaded into session
    for i in range(gen_iters):  # Fourth main() param
        # Format log output
        print("Story Number: {}".format(i + 1))
        print("CONTEXT: {}".format(contexts[i]))
        print("GIVEN ENDING: {}".format(endings[i]))

        # Instantiate empty data structs for sentence generation
        input_seq = np.array(sequences[i])[None]  # (1, n) input vector
        gen_end = []  # Stores each word of the generated endings
        word_prob = []  # Store probability distributions
        next_p = None  # Choose from random sample in probability space

        # Generate probability score for each word in our vocabulary
        for step in range(input_seq.shape[-1]):
            word_prob = rnn_lm.predict_on_batch(input_seq[:, step])[0, -1]

        # Sampling from word_prob, find probability of the next word
        while not gen_end or vocab_lookup[next_p][-1] not in end_of_sentence:
            next_p = np.random.choice(a=word_prob.shape[-1], p=word_prob)
            gen_end.append(next_p)
            word_prob = rnn_lm.predict_on_batch(
                np.array(next_p)[None, None])[0, -1]

        # Reset the hidden states to not influence the next ending
        rnn_lm.reset_states()

        # Finally go sequence to word from our vocab dictionary and print
        gen_end = ' '.join([vocab_lookup[word] for word in gen_end])
        print("GENERATED ENDING: {}\n".format(gen_end))


if __name__ == '__main__':
    vocab_path = 'data/roc_stories_full.csv'
    story_path = 'data/test_generation.csv'
    model_path = 'model/rnn_model.h5'
    gen_iters = 5
    main(model_path, vocab_path, story_path, gen_iters)
