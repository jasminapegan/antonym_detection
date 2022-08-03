import os
import random
import file_helpers


def create_val_test_set(in_data: str, in_examples: str, given_data: str, val_file: str, test_file: str, info_file: str,
                        sep: str, ratio: float=0.5, tmp_dir: str="tmp"):
    """
    Reads word data from 'in_data' and uses intersection/difference to words in 'given_data' to create validation and
    test set. Intersection words are divided according to 'ratio'. Default ratio will divide common data into equal
    number of sense data for test and validation. Senses present only in 'in_data' will become a part of 'val_file'.

    :param in_data : path to word data file containing '\t'-separated word data: word, location in sentence (index of
                     token), classification of word sense and sentence representing word sense
    :param in_examples : path to examples data file containing '\t'-separated word data: word, word form, sense label,
                    location in sentence (index of token) and sentence representing word sense
    :param given_data : path to word data file, containing '|'-separated data: word, POS, sense id, descriptor of sense
    :param val_file : output file location for validation data: '\t'-separated word, location of word, sentence
    :param test_file : output file location for test data: '\t'-separated word, location of word, sentence
    :param info_file : output file location for info about data
    :param sep : separator in given files
    :param ratio : validation to test ratio (default 0.5)
    :param tmp_dir : temporary files directory to store intersection, intersection part and difference data
                     (default "tmp")

    :return: None
    :returnType: None
    """

    intersection = os.path.join(tmp_dir, "intersection.txt")
    difference = os.path.join(tmp_dir, "difference.txt")

    file_helpers.filter_file_by_words(in_data, given_data, intersection, skip_idx=2, split_by=sep, split_by_2=sep) # skip classification
    file_helpers.filter_file_by_words(in_data, given_data, difference, skip_idx=2, split_by=sep, split_by_2=sep, complement=True)

    assert not file_helpers.is_empty_or_whitespace(intersection), "No words in %s and %s are common" % (in_data, given_data)

    pt1 = os.path.join(tmp_dir, "pt1.txt")

    intersection_examples = os.path.join(tmp_dir, "intersection_examples.txt")
    file_helpers.filter_file_by_words(in_examples, intersection, intersection_examples, split_by_2=sep)
    n_examples = file_helpers.file_len(intersection_examples)
    print(n_examples / 2, 0.001 * n_examples)

    i = n_examples
    while abs(i - n_examples/2) > 0.001 * n_examples:
        info_data = divide_word_senses(intersection, pt1, test_file, sep, ratio=ratio)

        test_examples_file = os.path.join(tmp_dir, "test.txt")
        file_helpers.filter_file_by_words(in_examples, test_file, test_examples_file, split_by_2=sep)
        test_examples = file_helpers.file_len(test_examples_file)

        i = test_examples
        print(i)

    file_helpers.concatenate_files([pt1, difference], val_file)
    val_examples_file = os.path.join(tmp_dir, "val.txt")
    file_helpers.filter_file_by_words(in_examples, val_file, val_examples_file, split_by_2=sep)
    val_examples = file_helpers.file_len(val_examples_file)

    with open(info_file, "w", encoding="utf8") as info:
        info.write(info_data)
        info.write("\n")
        val_words = list(file_helpers.count_words(val_file, sep=sep).items())
        val_senses = sum([w[1] for w in val_words])
        info.write("Validation file - words: %d senses: %d examples: %d\n" % (len(val_words), val_senses, val_examples))

        test_words = list(file_helpers.count_words(test_file, sep=sep).items())
        test_senses = sum([w[1] for w in test_words])
        info.write("Test file - words: %d senses: %d examples: %d\n" % (len(test_words), test_senses, test_examples))

def divide_word_senses(in_file: str, out_file1: str, out_file2: str, sep: str, ratio=0.5) -> str:
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
    words_count = list(file_helpers.count_words(in_file, sep=sep).items())
    random.shuffle(words_count)

    assert len(words_count) >= 2, "Cannot divide less than 2 words."

    sum_senses = 0
    idx = 0
    words_part = []

    while sum_senses < n_part:
        word, count = words_count[idx]
        words_part.append(word)
        sum_senses += count
        idx += 1

    with open(out_file1, "w", encoding="utf8") as out1:
        with open(out_file2, "w", encoding="utf8") as out2:
            with open(in_file, "r", encoding="utf8") as f:

                for line in f:
                    word = line.split(sep)[0]

                    if word in words_part:
                        out1.write(line)
                    else:
                        out2.write(line)

    n_words = len([w[1] for w in words_count])
    n_words_pt1 = len([w[1] for w in words_count[:idx]])
    n_words_pt2 = n_words - n_words_pt1

    info = "Words\tFile1: %d, File2: %d, total: %d\n" % (n_words_pt1, n_words_pt2, n_words)
    info += "Senses\tFile1: %d, File2: %d, total: %d\n" % (sum_senses, n-sum_senses, n)

    return info

def create_syn_ant_dataset(synonym_file, antonym_file, out_file, d=3, antonym_file2=None):
    with open(out_file, "w", encoding="utf8") as outf:

        visited = []
        with open(antonym_file, "r", encoding="utf8") as f:
            for line in f:
                data = line.split("\t")
                w1, w2, score = data[1], data[2], data[9]

                if score.count("d") > d:
                    visited.append((w1, w2))
                    outf.write(f"{w1}\t{w2}\t0\n")

        if antonym_file2:
            with open(antonym_file2, "r", encoding="utf8") as f:
                for line in f:
                    w1, w2 = line.strip().split("\t")

                    if (w1, w2) not in visited and (w2, w1) not in visited:
                        outf.write(f"{w1}\t{w2}\t0\n")

        with open(synonym_file, "r", encoding="utf8") as f:
            for line in f:
                w1, w2 = line.strip().split(" ")
                outf.write(f"{w1}\t{w2}\t1\n")

