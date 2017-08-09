import csv
import os


def csv2txt(read_path, write_path):
    txt_file = open('data/roc_text.txt', 'w')
    with open('data/example_train_stories.csv', 'r') as csv_file:
        story_reader = csv.reader(csv_file, delimiter=',')
        story_lists = [row for row in story_reader]
        stories = [' '.join(story) for story in story_lists]
        [txt_file.write(story + '\n') for story in stories]


def main(read_path, write_path):
    # generate text file for char-rnn language model
    csv2txt(read_path, write_path)


if __name__ == '__main__':
    csv_file = os.path.join('data', 'example_train_stories.csv')
    txt_file = os.path.join('data', 'roc_text.txt')
    print('\nReading Data from {}\nWriting to {}\n'.format(csv_file, txt_file))
    main(csv_file, txt_file)
