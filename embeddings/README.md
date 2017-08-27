## Setup

For dependencies, run `pip install -r requirements.txt`

The main one is obviously spaCy. I ran these scripts in a virtual environment. While its not necessary for local development, when you integrate this into the application stack, containerizing the service might be ideal.

You can largely ignore the test script.

## Usage

Really, it just comes down to the example three lines of code in the `main()` function.

1. Process each word through the spaCy NLP pipeline
2. Retrieve the spaCy token objects from their parent doc objects
3. Compute the cosine similarity metric between each word embedding

I documented the code, which is just a wrapper for the spaCy API. In terms of implementation, loading the spaCy english model each time the function is called will add a lot of overhead. Ideally, this would be loaded once when the application boots.

`get_vec(a)` is an additional function which will fetch the actual word embedding of a spaCy token. Useful if you want to store the GLoVe embeddings.