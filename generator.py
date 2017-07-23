import csv
import pickle
import numpy as np


def main():
    # Lod tokenizer
    with open('tokenizer_96000.pkl', 'rb') as pklfile:
        token = pickle.load(pklfile)
        print("Vocab size: {}".format(len(token.word_index)))

    # # Load 25 contexts and endings
    with open('data/test_generation.csv', 'r') as csvfile:
        test_stories = [story for story in csv.reader(csvfile)]
    endings = [story[-1] for story in test_stories[-20:]]
    contexts = [' '.join(story[:-1]) for story in test_stories[-20:]]
    indexes = token.texts_to_sequences(contexts)
    print("First Context: {}\n{}".format(contexts[0], indexes[0]))
    print("First Ending: {}".format(endings[0]))

    # # Create a lookup table for the vocab indexes
    # vocab_lookup = {index: word for word, index in token.word_index.items()}
    # end_of_sentence = ['.', '?', '!']
    # pp.pprint(list(vocab_lookup.items())[:20])

    # # Finally, generate some endings given a context!
    # for story, story_idxs, ending in zip(contexts, indexes, endings):
    #     print("Context: ", story)
    #     print("Given Ending: ", ending)
    #     generated_ending = []
    #     story_idxs = np.array(story_idxs)[None]
    #     for step_idx in range(story_idxs.shape[-1]):
    #         # Predict probability of next word
    #         next_word = rnn_model.predict_on_batch(
    #             story_idxs[:, step_idx])[0, -1]
    #     while not generated_ending or vocab_lookup[next_word][-1] not in end_of_sentence:
    #         next_word = np.random.choice(a=next_word.shape[-1], p=next_word)
    #         generated_ending.append(next_word)
    #         next_word = rnn_model.predict_on_batch(
    #             np.array(next_word)[None, None])[0, -1]
    #     rnn_model.reset_states()
    #     generated_ending = ' '.join([vocab_lookup[word]
    #                                  for word in generated_ending])
    #     print("Generated Ending: {}\n".format(generated_ending))


if __name__ == '__main__':
    main()
