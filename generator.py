import csv
import pickle
import pprint as pp
# import numpy as np

from keras.models import Sequential
from keras.layers import Dense, TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU


def build_rnn(vocab_size, embeddings, hidden, batch_size, timesteps):
    model = Sequential()
    model.add(Embedding(batch_input_shape=(batch_size, timesteps),
                        input_dim=vocab_size + 1,
                        output_dim=embeddings,
                        mask_zero=True))
    model.add(GRU(hidden,
                  return_sequences=True,
                  stateful=True))
    model.add(GRU(hidden,
                  return_sequences=True,
                  stateful=True))
    model.add(TimeDistributed(Dense(vocab_size + 1, activation='softmax')))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model


def main():
    # Load 25 stories to test generative story endings
    with open('data/test_generation.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        test_stories = []
        for story in reader:
            # Test file has a blank column, just remove it if it exists
            if len(story) == 6:
                del story[-1]
                test_stories.append(story)
            else:
                test_stories.append(story)
    print("Will generate {} endings.".format(len(test_stories)))

    # Load larger vocabulary from different tokenizer
    with open('tokenizer_96000.pkl', 'rb') as pklfile:
        token = pickle.load(pklfile, encoding='latin1')
        print("Vocab size: {}".format(len(token.word_index)))

    # Prepare inputs for the sequence generator
    endings = [story[-1] for story in test_stories]  # Labels for each story
    contexts = [' '.join(story[:-1]) for story in test_stories]  # Stories
    context_idxs = token.texts_to_sequences(contexts)  # Input vecs to the RNN
    pp.pprint(contexts[0])
    pp.pprint(context_idxs[0])
    pp.pprint(endings[0])

    # Create a lookup table for the vocab indexes
    vocab_lookup = {index: word for word, index in token.word_index.items()}
    end_of_sentence = ['.', '?', '!']  # eos tokens
    print("Generated sentences will end on a", end_of_sentence)
    pp.pprint(list(vocab_lookup.items())[:10])

    # Compile RNN architecture
    vocab_size = len(token.word_index)
    embeddings = 300
    hidden_cells = 500
    batch_size = 1
    steps = 1
    print("Compiling model with the following hyperparams:\n")
    print(
        "Vocab: {}\nEmbeddings: {}\nHidden: {}\nBatches: {}\nSteps: {}".format(
            vocab_size, embeddings, hidden_cells, batch_size, steps))
    rnn_model = build_rnn(vocab_size=vocab_size,
                          embeddings=embeddings,
                          hidden=hidden_cells,
                          batch_size=batch_size,
                          timesteps=steps)
    print("Model Compiled...")

    # Load pre-trained weights
    print("Loading Trained Weights...")
    rnn_model.load_weights('rnn_weights_96000.h5')
    print("Weights Trained Loaded...")


if __name__ == '__main__':
    main()
