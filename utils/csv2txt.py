import csv
import os


def csv2txt(read_path, write_path):
    txt_file = open(write_path, 'w')
    with open(read_path, 'r') as csv_file:
        story_reader = csv.reader(csv_file, delimiter=',')
        story_lists = [row for row in story_reader]
        stories = [' '.join(story) for story in story_lists]
        [txt_file.write(story + '\n') for story in stories]


def main(read_path, write_path):
    # generate text file for char-rnn language model
    csv2txt(read_path, write_path)


if __name__ == '__main__':
    csv_file = os.path.join('data', 'roc_stories_full.csv')
    txt_file = os.path.join('data', 'roc_text_full.txt')
    print('\nReading Data from {}\nWriting to {}\n'.format(
        csv_file, txt_file)
    )
    main(csv_file, txt_file)
