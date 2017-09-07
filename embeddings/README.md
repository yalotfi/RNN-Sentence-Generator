## Using Word Embeddings

Representing words as dense vectors proves useful both on their own and as inputs for generative language models.

### Global Vector for Word Representation (GLoVe)

An unsupervised model that came out of the Stanford NLP group. It is trained on a co-occurence matrix which counts word-word frequencies in a corpus. For our purposes, the spaCy NLP module provides high level API methods to access pre-trained word vectors.

### Word2Vec

Another model called Word2Vec takes a neuronal approach to learning vector representations of words. The objective task is defined by either the skip-gram or continuous bag of words models wherein the model predicts a word given a context or predicts a context given a word, respectively. In this case, the gensim Python package provides a highly optimized solution for training a Wrod2Vec model.