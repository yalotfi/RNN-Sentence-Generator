import csv
import pprint as pp
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.models import load_model
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


def load_corpus(read):
    with open(read, 'r') as csvfile:
        story_reader = csv.reader(csvfile, delimiter=',')
        story_lists = [row for row in story_reader]
    return [' '.join(story) for story in story_lists]


def main(vocab_path, story_path, model_path):
    # Load and process ROC corpus
    corpus = load_corpus(vocab_path)  # First main() param

    # Use Keras primitive to tokenize ROC corpus
    token = Tokenizer(lower=True, filters='')
    token.fit_on_texts(corpus)
    print("\nVocab size: {}\n".format(len(token.word_index)))

    # Create a lookup table for the vocab indexes
    vocab_lookup = {index: word for word, index in token.word_index.items()}
    end_of_sentence = ['.', '?', '!']  # eos tokens

    # Load 25 stories to test generative story endings
    with open(story_path, 'r') as csvfile:  # Second main() param
        reader = csv.reader(csvfile)
        test_stories = []
        for story in reader:
            # Test file has a blank column, just remove it if it exists
            if len(story) == 6:
                del story[-1]
                test_stories.append(story)
            else:
                test_stories.append(story)
    print("Will generate {} endings.\n".format(len(test_stories)))

    contexts, endings = [], []
    for story in test_stories:
        contexts.append(' '.join(story[:-1]))
        endings.append(story[-1])
    sequences = token.texts_to_sequences(contexts)
    print("CONTEXT:\n{}\n".format(contexts[0]))
    print("INPUT SEQUENCE:\n{}\n".format(sequences[0]))
    print("ENDING:\n{}\n".format(endings[0]))

    ''' TESTING '''
    # Load trained RNN model
    print("\nLoading Model...\n")
    rnn_lm = load_model(model_path)  # Third main() param
    print("\nModel Ready...\n")

    input_seq = np.array(sequences[0])[None]  # (1, n) input vector

    # print(input_seq)
    # print(input_seq[:, 2])
    # print(input_seq[:, 2].shape)

    # word_prob = rnn_lm.predict_on_batch(input_seq[:, 0])
    # print(word_prob)

    generated_ending = []
    word_prob = []  # Store probability distributions
    for step in range(input_seq.shape[-1]):
        word_prob = rnn_lm.predict_on_batch(input_seq[:, step])[0, -1]
    while not generated_ending or vocab_lookup[next_word][-1] not in end_of_sentence:
        next_word = np.random.choice(a=word_prob.shape[-1], p=word_prob)
        generated_ending.append(next_word)
        word_prob = rnn_lm.predict_on_batch(
            np.array(next_word)[None, None])[0, -1]
    prob_dist = generated_ending
    generated_ending = ' '.join([vocab_lookup[word]
                                 for word in generated_ending])
    print("Highest Probabilities: {}\n".format(prob_dist))
    print("GENERATED ENDING: {}\n".format(generated_ending))
    ''' TESTING '''

    # # Generate each word of the fifth sentence until it hits an eos tag
    # for context, idxs, ending in zip(contexts, context_idxs, endings):
    #     print("STORY: ", context)
    #     print("GIVEN ENDING: ", ending)
    #     generated_ending = []
    #     idxs = np.array(idxs)[None]
    #     for step_idx in range(idxs.shape[-1]):
    #         word_prob = rnn_lm.predict_on_batch(step_idx[:, step_idx])[0, -1]
    #     while not generated_ending or vocab_lookup[next_word][-1] not in end_of_sentence:
    #         next_word = np.random.choice(a=word_prob.shape[-1], p=word_prob)
    #         generated_ending.append(next_word)
    #         word_prob = rnn_lm.predict_on_batch(
    #             np.array(next_word)[None, None])[0, -1]
    #     rnn_lm.reset_states()
    #     generated_ending = ' '.join([vocab_lookup[word]
    #                                  for word in generated_ending])
    #     print("GENERATED ENDING: {}\n".format(generated_ending))

    # # Compile RNN architecture
    # vocab_size = len(token.word_index)
    # embeddings = 300
    # hidden_cells = 500
    # batch_size = 1
    # steps = 1
    # print("\nCompiling model with the following hyperparams:")
    # print(
    #     "Vocab: {}\nEmbeddings: {}\nHidden: {}\nBatches: {}\nSteps: {}".format(
    #         vocab_size, embeddings, hidden_cells, batch_size, steps))
    # rnn_model = build_rnn(vocab_size=vocab_size,
    #                       embeddings=embeddings,
    #                       hidden=hidden_cells,
    #                       batch_size=batch_size,
    #                       timesteps=steps)
    # print("\nModel Compiled...\n")

    # # Load pre-trained weights
    # print("Loading Trained Weights...\n")
    # rnn_model.load_weights(weights_path)  # Third main() param
    # print("Weights Trained Loaded...\n")


if __name__ == '__main__':
    vocab_path = 'data/roc_stories_full.csv'
    story_path = 'data/test_generation.csv'
    model_path = 'model/rnn_model.h5'
    main(vocab_path, story_path, model_path)
