import os
import file_helpers


def create_val_test_set(in_data: str,
                        given_data: str,
                        val_file: str,
                        test_file: str,
                        ratio: 0.5,
                        tmp_dir: "tmp"):
    """
    Reads word data from 'in_data' and uses intersection/difference to words in 'given_data' to create validation and
    test set. Intersection words are divided according to 'ratio'. Default ratio will divide common data into equal
    number of sense data for test and validation. Senses present only in 'in_data' will become a part of 'val_file'.

    :param in_data : path to word data file containing '\t'-separated word data: word, location in sentence (index of
                     token), classification of word sense and sentence representing word sense
    :param given_data : path to word data file, containing '|'-separated data: word, POS, sense id, descriptor of sense
    :param val_file : output file location for validation data: '\t'-separated word, location of word, sentence
    :param test_file : output file location for test data: '\t'-separated word, location of word, sentence
    :param ratio : validation to test ratio (default 0.5)
    :param tmp_dir : temporary files directory to store intersection, intersection part and difference data
                     (default "tmp")

    :return: None
    :returnType: None
    """

    intersection = os.path.join(tmp_dir, "intersection.txt")
    difference = os.path.join(tmp_dir, "difference.txt")

    file_helpers.filter_file_by_words(in_data, given_data, intersection, skip_idx=1)    # skip classification
    file_helpers.filter_file_by_words(in_data, given_data, difference, skip_idx=1, complement=True)

    assert not file_helpers.is_empty(intersection), "No words in %s and %s are common" % (in_data, given_data)

    pt1 = os.path.join(tmp_dir, "pt1.txt")
    divide_word_senses(intersection, pt1, test_file, ratio=ratio)

    file_helpers.concatenate_files([pt1, difference], val_file)

def divide_word_senses(in_file: str, out_file1: str, out_file2: str, ratio=0.5):
    """
    Divides word senses found in 'in_file' into 'out_file1' and 'out_file2' according to 'ratio'. Senses of the same
    word will not be divided. Because of that, ratio will not necessarily be exactly reached. Atleast 'ratio' of word
    senses will be in 'out_file1'.

    :param in_file: path to file containing word senses data
    :param out_file1: path to file into which cca. 'ratio' part of senses will be saved.
    :param out_file2: path to file into which cca. 1 - 'ratio' part of senses will be saved.
    :param ratio: the ratio by which senses are divided
    :return: None
    :raises ValueError: if ratio is not element of the (0, 1) interval
    :raises AssertionError: if 'in_file' contains senses for less than 2 words
    """

    if not 0 < ratio < 1:
        raise ValueError("Ratio must lay in interval (0, 1).")

    n = file_helpers.file_len(in_file)
    n_part = n * ratio

    words_data = file_helpers.load_validation_file_grouped(in_file, all_strings=True)
    words_count = [(key, len(words_data[key]['sentences'])) for key in words_data.keys()]
    file_helpers.shuffle(words_count)

    assert len(words_count) >= 2, "Cannot divide less than 2 words."

    sum = 0
    idx = 0
    words_part = []

    while sum < n_part:
        word, count = words_count[idx]
        words_part.append(word)
        sum += count
        idx += 1

    print("File1: %d, File2: %d, total: %d" % (sum, n-sum, n))

    with open(out_file1, "w", encoding="utf8") as out1:
        with open(out_file2, "w", encoding="utf8") as out2:
            with open(in_file, "r", encoding="utf8") as f:

                for line in f:
                    word = line.split("\t")[0]

                    if word in words_part:
                        out1.write(line)
                    else:
                        out2.write(line)
