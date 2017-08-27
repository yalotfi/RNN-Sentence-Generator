import en_core_web_sm


def get_doc(nlp, a):
    """
    Description:
        Process text through the spaCy NLP pipeline.

    Arguments:
        nlp - Take the spaCy NLP object which is instantiated when the English
              model is loaded separate from when this function is called.
        a - word to be processed

    Return:
        doc - spaCy doc object
    """
    return nlp(a)


def get_token(a):
    """
    Return spaCy token object from parent doc object with a strict
    assumption that only a single word is in the doc
    """
    assert(len(a) == 1)
    return a[0]


def get_vec(a):
    """
    Return the raw GLoVe embedding of a spaCy token object which is a vector
    of shape (300, 1)
    """
    assert(a.vector.shape == (300,))
    return a.vector.reshape((300, 1))


def semantic_score(a, b):
    """
    Description:
        Computes cosine similarity between two GLoVe embeddings. Pre-trained
        word embeddings are vectors with shape (300, 1) stored as properties
        of spaCy's token object.

    Arguments:
        a - first spaCy token object with parent doc object
        b - second spaCy token object with parent doc object

    Returns:
        score - float32 score of semantic similarity or zero if the embedding
                does not exist
    """
    if a.has_vector and b.has_vector:
        return a.similarity(b)
    else:
        return 0


def main():
    ############################
    # LOAD AT START OF SESSION #
    ############################
    nlp = en_core_web_sm.load()
    ############################
    test_words = ['river', 'fish']

    # Example on using the above functions
    docs = [get_doc(nlp, word) for word in test_words]
    tokens = [get_token(doc) for doc in docs]
    score = semantic_score(tokens[0], tokens[1])

    # Log the output score when script is run
    print(score)


if __name__ == '__main__':
    main()
