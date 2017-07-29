# Purpose:

Train a Reccurent Neural Network Language Model that is able to generate a fifth sentence based on the preceding four sentences of a short story.

# To Do's:

1. GPU-Enabled Workstation
    * Install cuDNN, Tensorflow, etc on local machine to save on AWS pricing and not rely on pre-trained RNNs
    * Re-train this RNN on my GTX970 for 25 epochs and compare

2. Tokenization
    * Utilize a more sophisticated tokenizer from NLTK or SpaCy
    * Should produce more syntactically correct sentences because it recognizes special characters, unlike the Keras text preprocesser.

3. Word Embeddings
    * Incorporate GLoVe or Word2Vec embeddings in the model

4. Encoder-Decoder w/ Attention Mechanism Architecture
    * State-of-the-art Language Models are using attention mechanisms
    * Instead of encoding full source text, decoder will focus on parts of the sentence and learn where to focus.
    * Improved representation of long-term dependencies