import csv
import pprint as pp
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
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


def load_stories(read):
    with open(read, 'r') as csvfile:
        story_reader = csv.reader(csvfile, delimiter=',')
        story_lists = [row for row in story_reader]
    return [' '.join(story) for story in story_lists]


def main(read, train=False):
    stories = load_stories(read)  # Load ROC dataset

    # Process story data with Keras tokenizer
    token = Tokenizer()
    token.fit_on_texts(stories)  # Splits stories into indexed words
    vocab_idxs = token.texts_to_sequences(stories)  # Vectorize story indexes
    max_len = max([len(story) for story in vocab_idxs])  # Longest story
    # Create matrix, X, where X[i, :] is a story vector
    story_matrix = pad_sequences(vocab_idxs, maxlen=max_len)

    # Offset each word by 1 so that every input word, x[i], maps to y[i + 1]
    X_train, y_train = story_matrix[:, :-1], story_matrix[:, 1:]

    # Graph RNN with input hyperparams
    print("Building Model...")
    batch_size = 20
    rnn_model = build_rnn(vocab_size=len(token.word_index),
                          embeddings=300,
                          hidden=500,
                          batch_size=batch_size,
                          timesteps=max_len - 1)
    print("Model Compiled...")

    # Either train the model or load weights of a trained RNN
    if train:
        # Train the RNN
        epochs = 10
        print("Training RNN on {} stories for {} epochs...".format(
            len(X_train), epochs)
        )
        for epoch in range(epochs):
            epoch_loss = []
            for batch in range(0, len(X_train)):
                batch_x = X_train[batch: batch + batch_size]
                batch_y = y_train[batch: batch + batch_size, :, None]
                batch_loss = rnn_model.train_on_batch(batch_x, batch_y)
                epoch_loss.append(batch_loss)
            print("Epoch: {} | Mean Error {%.3f}".format(
                epoch + 1, np.mean(epoch_loss))
            )
            rnn_model.save_weights('rnn_weights.h5')
    else:
        # Load model weights
        print("Loading Weights...")
        rnn_model.load_weights('rnn_weights_96000.h5')
        print("Weights Loaded...")

    # Load 25 contexts and endings
    with open('data/test_generation.csv', 'r') as csvfile:
        test_stories = [story for story in csv.reader(csvfile)]
    endings = [story[-1] for story in test_stories[-10:]]
    contexts = [' '.join(story[:-1]) for story in test_stories[-10:]]
    indexes = token.texts_to_sequences(contexts)
    print("First Context: {}\n{}".format(contexts[0], indexes[0]))
    print("First Ending: {}".format(endings[0]))

    # Create a lookup table for the vocab indexes
    vocab_lookup = {index: word for word, index in token.word_index.items()}
    end_of_sentence = ['.', '?', '!']
    pp.pprint(list(vocab_lookup.items())[:20])

    # Finally, generate some endings given a context!
    for story, story_idxs, ending in zip(contexts, indexes, endings):
        print("Context: ", story)
        print("Given Ending: ", ending)
        generated_ending = []
        story_idxs = np.array(story_idxs)[None]
        for step_idx in range(story_idxs.shape[-1]):
            # Predict probability of next word
            next_word = rnn_model.predict_on_batch(
                story_idxs[:, step_idx])[0, -1]
        while not generated_ending or vocab_lookup[next_word][-1] not in end_of_sentence:
            next_word = np.random.choice(a=next_word.shape[-1], p=next_word)
            generated_ending.append(next_word)
            next_word = rnn_model.predict_on_batch(
                np.array(next_word)[None, None])[0, -1]
        rnn_model.reset_states()
        generated_ending = ' '.join([vocab_lookup[word]
                                     for word in generated_ending])
        print("Generated Ending: {}\n".format(generated_ending))


if __name__ == '__main__':
    main('data/roc_stories_full.csv', train=False)
