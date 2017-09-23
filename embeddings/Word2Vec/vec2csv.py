import os
import csv

from gensim.models import Word2Vec


def load_targets(fpath):
    """Return a list of target words from CSV file"""
    with open(fpath, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        return [row[0] for row in reader]


def load_vecs(fpath):
    """ Return a trained Word2Vec model"""
    return Word2Vec.load(fpath)


def most_similar(model, word, topn=10):
    return model.similar_by_word(word, topn=topn)


def vec2csv(model, targets, outfile, topn=10):
    with open(outfile, 'w', newline='') as f:
        writer = csv.writer(f, dialect='excel')
        for target in targets:
            similarities = most_similar(model, target, topn=topn)
            for (word, score) in similarities:
                row = [target, word, score]
                writer.writerow(row)


def main(infile, outfile, model_name, n=10):
    # Set correct path for file i/o
    inpath = os.path.join('embeddings', 'Word2Vec', infile)
    outpath = os.path.join('embeddings', 'Word2Vec', outfile)
    vecpath = os.path.join('models', 'embeddings', model_name)

    # Load infile and model
    targets = load_targets(inpath)
    model = load_vecs(vecpath)

    # Write scores to outfile
    vec2csv(model, targets, outpath, n)


if __name__ == '__main__':
    infile = "test_targets.csv"
    outfile = "import_scores.csv"
    model_name = "vecs_300_window11"
    n_word = 10
    main(infile, outfile, model_name, n_word)
