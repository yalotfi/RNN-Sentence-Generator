from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras.optimizers import RMSprop


def lstm_basic(hidden, max_len, vocab_size):
    model = Sequential()
    model.add(LSTM(hidden, input_shape=(max_len, vocab_size)))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
    return model


def lstm_embedding(batch_size, timesteps, vocab_size, embeddings, hidden):
    print("Building Model...")
    model = Sequential()
    model.add(Embedding(batch_input_shape=(batch_size, timesteps),
                        input_dim=vocab_size + 1,
                        output_dim=embeddings,
                        mask_zero=True))
    model.add(LSTM(hidden, return_sequences=True, stateful=True))
    model.add(Dense(vocab_size + 1, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=RMSprop(lr=0.01))
    return model
