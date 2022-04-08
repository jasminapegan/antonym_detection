import os
import pickle
from io import TextIOWrapper

import numpy as np
from typing import List, Dict, Union


def get_words_data_from_file(words_file: str, sep='|', header=True, skip_num=False) -> List[Dict]:
    """
    Parse data from 'words_file' into a list of dicts

    :param words_file: file containing words data
    :param sep: separator in words_file (default '|')
    :return: a list of dictionaries containing word data: word, word type, sense id, description of sense
    """

    words = []

    with open(words_file, "r", encoding="utf8") as in_file:
        for i, line in enumerate(in_file):

            # skip header
            if not header or i != 0:
                if skip_num:
                    num_of_meaning = None
                    word, word_type, description = line.strip().split(sep)
                else:
                    word, word_type, num_of_meaning, description = line.strip().split(sep)
                words.append({"word": word,
                              "type": word_type,
                              "num": num_of_meaning,
                              "description": description})
    return words

def words_data_to_dict(words_file: str, sep='|', header=True, skip_num=True) -> Dict:
    """
    Read word data from file and convert a list of dictionaries with sense data to a dictionary of words data

    :param words_file: file containing words data
    :param sep: separator in words_file (default '|')
    :return: a dictionary with words as keys and values as list of {word, type, sense_id, description}
    """

    words = get_words_data_from_file(words_file, sep=sep, header=header, skip_num=skip_num)
    words_dict = {}

    for word_data in words:
        word = word_data["word"]
        if word not in words_dict.keys():
            words_dict[word] = [word_data]
        else:
            words_dict[word].append(word_data)
    return words_dict

def save_json_word_data(words_file: str, out_file: str):
    """
    Get words data from 'words_file', convert to json form and save to binary 'out_file'.

    :param words_file: file containing words data
    :param out_file: file to save binary json to
    :return: None
    """

    words_dict = words_data_to_dict(words_file)

    with open(out_file, "wb") as outjson:
        pickle.dump(words_dict, outjson)

def save_json_word_data_from_multiple(words_file1: str, words_file2: str, out_file: str):
    words_dict1 = words_data_to_dict(words_file1)
    words_dict2 = words_data_to_dict(words_file2)

    for word in words_dict2.keys():

        if word not in words_dict1.keys():
            words_dict1[word] = words_dict2[word]

        else:
            n1 = len(words_dict1[word])
            n2 = len(words_dict2[word])

            if n1 < n2:
                words_dict1[word] = words_dict2[word]

    with open(out_file, "wb") as outjson:
        pickle.dump(words_dict1, outjson)

def load_json_word_data(json_file: str):
    with open(json_file, "rb") as words_dict_json:
        return pickle.load(words_dict_json)

def get_unique_words(word_file: str, sep: str='|'):
    return list(set([line[0] for line in load_file(word_file, sep=sep)]))

def load_file(file: str, limit: int=None, sep: str='\t', skip_header: bool=False):
    """
    Read data from 'file', return a list of lines (string lists).

    :param file: file with data to read
    :param limit: until which line to read. If None, read all lines (default None)
    :param sep: separator of 'file'
    :return: a list of lists of string representing lines of data
    """

    data = []
    with open(file, "r", encoding="utf8") as f:

        for i, line in enumerate(f):
            if i == 0 and skip_header:
                continue

            data.append(line.strip().split(sep))

            if i == limit:
                return data

    return data

def is_empty_or_whitespace(filename: str):
    # check if file is empty or contains only whitespace
    with open(filename, encoding='utf8') as f:
        for line in f:
            if line.strip() != "":
                return False
    return True

def file_len(filename: str):
    # return length of the file
    i = 0
    with open(filename, encoding='utf8') as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def load_validation_file_grouped(file: str, all_strings: bool=False, indices: bool=False) -> Dict[str, Dict]:
    """
    Parses data in 'file' and returns a dictionary of parsed data per word.

    :param file: input tsv data file
    :param all_strings: output split data with no conversion? (default False)
    :param indices: does 'file' contain index of the word in sentence data? (default True)
    :return: dictionary of data per word (dict: indices, embeddings, sentences, labels)
    """

    data = load_file(file, sep='\t')
    words_data = {}

    for line in data:

        word, word_form, label, index, sentence = line
        index = None

        if indices:
            index = line[-2]
            if not all_strings:
                index = int(line[-2])

        if word in words_data.keys():
            if indices:
                words_data[word]['indices'].append(index)
            words_data[word]['labels'].append(label)
            words_data[word]['sentences'].append(sentence)

        else:
            words_data[word] = {'labels': [label], 'sentences': [sentence]}
            if indices:
                words_data[word]['indices'] = [index]

    return words_data

def load_result_file_grouped(file: str, embeddings: bool=False, skip_header: bool=False) -> Dict[str, Dict]:
    """
    Parses data in 'file' and returns a dictionary of parsed data per word.

    :param file: input tsv data file
    :param embeddings: does 'file' contain embeddings? (default False)
    :return: dictionary of data per word (dict: cluster labels, sentences, embeddings)
    """

    data = load_file(file, sep='\t')
    words_data = {}

    for i, line in enumerate(data):
        if skip_header and i == 0:
            continue

        if embeddings:
            label, word, sentence, embedding = line
        else:
            label, word, sentence = line

        if word in words_data.keys():
            words_data[word]['labels'].append(label)
            words_data[word]['sentences'].append(sentence)

            if embeddings:
                words_data[word]['embeddings'].append(embedding)

        else:
            words_data[word] = {'labels': [label],
                                'sentences': [sentence],
                                'embeddings': [embedding]}

    return words_data

def write_word_data(out_file: str, word_data: Dict[str, List[Dict]], sep='|'):
    with open(out_file, "w", encoding="utf8") as f:
        for word, data_list in sorted(list(word_data.items())):
            for data in data_list:
               f.write(sep.join([word, data['type'], data['num'], data['description']]))

def write_grouped_data(outf: TextIOWrapper, data: List, centroid: List=None):
    for label, word, sentence, embedding in data: #

        if centroid is not None:
            centroid_string = " ".join([str(x) for x in centroid])
            outf.write("\t".join([str(label), label, centroid_string]) + "\n")

        #embedding_str = " ".join([str(x) for x in embedding])
        outf.write("\t".join([str(label), word, sentence]) + "\n") #, embedding_str

def write_data_for_classification(outf: TextIOWrapper, data: List):
    for label, word, centroid in data:
        centroid_string = " ".join([str(x) for x in centroid])
        outf.write("\t".join([str(label), word, centroid_string]) + "\n")

def count_words(in_file: str, sep='\t') -> Dict[str, int]:
    """
    Read data from 'in_file' and return dictionary of word counts.

    :param in_file: file containing word data
    :param sep: separator of 'in_file' (default '\t')
    :return: dictionary of word counts
    """
    words = {}

    with open(in_file, 'r', encoding="utf8") as input:
        lines = input.readlines()

        for line in lines:
            word = line.split(sep)[0]

            if word in words.keys():
                words[word] += 1
            else:
                words[word] = 1

    return words

def filter_file_by_words(file: str, words_file_or_list: Union[str, List[str]], out_file: str, word_idx: int=0,
                         split_by: str='\t', complement: bool=False, skip_idx: int=None):
    """
    Filter 'file' by words in 'word_file'. Output lines starting with words in 'word_file' to 'out_file'.

    :param file: base file to be filtered
    :param words_file: containing words to filter by
    :param out_file: file to output filtered data
    :param word_idx: index of word in 'words_file' (default 0)
    :param split_by: separator in 'file' (default '\t')
    :param complement: if True, keep lines matching filter. If False, filter out these lines (default True)
    :param skip_idx: if set to integer, skips an index in each line to write to output (default None)
    :return: None
    """

    if isinstance(words_file_or_list, str):
        words = get_unique_words(words_file_or_list)
    else:
        words = words_file_or_list

    with open(file, "r", encoding="utf8") as f:
        with open(out_file, "w", encoding="utf8") as outf:

            for line in f:
                word = line.split(split_by)[word_idx]

                if (complement and word not in words) or (not complement and word in words):
                    if skip_idx:
                        split_line = line.split(split_by)
                        out_line = split_by.join([x for i, x in enumerate(split_line) if i != skip_idx])

                        if out_line[-1] != "\n":
                            out_line += "\n"

                        outf.write(out_line)
                    else:
                        outf.write(line)

def concatenate_files(file_list: List[str], out_file: str):
    # write content of files in 'file_list' to 'out_file'
    with open(out_file, "w", encoding="utf8") as out:
        for f in file_list:
            with open(f, "r", encoding="utf8") as in_file:
                out.writelines(in_file.readlines())

def remove_duplicate_lines(in_file: str, out_file: str, range: int=None):
    # copy 'in_file' to 'out_file' skipping lines that repeat in last 'range' lines
    visited_lines = []

    with open(in_file, 'r', encoding="utf8") as input:
        with open(out_file, 'wt', encoding="utf8") as output:

            for line in input.readlines():
                if line not in visited_lines:

                    output.write(line)
                    visited_lines.append(line)
                    if range:
                        visited_lines = visited_lines[-range:]

def sort_lines(in_file: str, out_file: str, sep='\t'):
    # sort lines by element at 0

    with open(in_file, 'r', encoding="utf8") as input:
        with open(out_file, 'wt', encoding="utf8") as output:

            lines = input.readlines()
            lines.sort(key=lambda x: x.split(sep)[0])

            for line in lines:
                output.write(line)

def convert_to_np_array(string_list: List[str]):
    return np.array([x.split(' ') for x in string_list], dtype=float)

def count_lines(folder: str):
    return sum([file_len(folder + "/" + file) for file in os.listdir(folder)])

def get_all_words(words_file_list, words_file_source, out_file, tmp_dir='tmp'):
    tmp_all = os.path.join(tmp_dir, "all_words")
    with open(tmp_all, "w", encoding="utf8") as outf:

        for words_file in words_file_list:
            with open(words_file, "r", encoding="utf8") as f:
                for line in f:
                    outf.write(line)

        with open(words_file_source, "r", encoding="utf8") as f:
            for line in f:
                word, pos, label, description = line.split("|")
                outf.write("|".join([word, pos, description]))

    tmp_sorted = os.path.join(tmp_dir, "all_sorted_words.txt")
    sort_lines(tmp_all, tmp_sorted, sep='|')
    remove_duplicate_lines(tmp_sorted, out_file, range=1)
