import hashlib
import os
import pickle
from datetime import datetime
from difflib import get_close_matches
from io import TextIOWrapper

import classla
import numpy as np
from typing import List, Dict, Union

from data.lemmatization import get_sentence_lemmas_list


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

def get_unique_words_pos(word_file: str, sep: str='|'):
    words = {}
    for line in load_file(word_file, sep=sep):
        if len(line) < 2:
            continue

        word, pos = line[:2]

        if word not in words:
            words[word] = [pos]
        elif pos not in words[word]:
            words[word].append(pos)

    return words

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

def load_validation_file_grouped(file: str, all_strings: bool=False, indices: bool=False, embeddings: bool=False,
                                 sep='\t', use_pos=True) -> Dict[str, Dict]:
    """
    Parses data in 'file' and returns a dictionary of parsed data per word.

    :param file: input tsv data file
    :param all_strings: output split data with no conversion? (default False)
    :param indices: does 'file' contain index of the word in sentence data? (default True)
    :return: dictionary of data per word (dict: indices, embeddings, sentences, labels)
    """

    data = load_file(file, sep=sep)
    words_data = {}

    for line in data:

        if use_pos:
            if embeddings:
                word, pos, label, sentence, embedding = line
            else:
                word, pos, word_form, label, index, sentence = line
        else:
            if embeddings:
                word, label, sentence, embedding = line
            else:
                word, word_form, label, index, sentence = line
            pos = "all"

        index = None

        if indices:
            index = line[-2]
            if not all_strings:
                index = int(line[-2])

        if word in words_data.keys():
            if pos in words_data[word].keys():

                if indices:
                    words_data[word][pos]['indices'].append(index)
                if embeddings:
                    words_data[word][pos]['embeddings'].append(embedding)

                words_data[word][pos]['labels'].append(label)
                words_data[word][pos]['sentences'].append(sentence)

            else:
                words_data[word][pos] = {'labels': [label], 'sentences': [sentence]}
                if indices:
                    words_data[word][pos]['indices'] = [index]
                if embeddings:
                    words_data[word][pos]['embeddings'] = [embedding]

        else:
            words_data[word] = {pos: {'labels': [label], 'sentences': [sentence]}}
            if indices:
                words_data[word][pos]['indices'] = [index]
            if embeddings:
                words_data[word][pos]['embeddings'] = [embedding]

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
            if embeddings:
                words_data[word] = {'labels': [label],
                                    'sentences': [sentence],
                                    'embeddings': [embedding]}
            else:
                words_data[word] = {'labels': [label],
                                    'sentences': [sentence]}

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

def count_words(in_file: str, sep='\t', indices=[0]) -> Dict[str, int]:
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
            new_words = [x for i, x in enumerate(line.split(sep)) if i in indices]

            for word in new_words:
                if word in words.keys():
                    words[word] += 1
                else:
                    words[word] = 1

    return words

def filter_file_by_words(file: str, words_file_or_list: Union[str, List[str]], out_file: str, word_idx: int=0,
                         split_by: str='\t', split_by_2:str='\t', complement: bool=False, skip_idx: int=None):
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
        words = get_unique_words(words_file_or_list, sep=split_by_2)
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
            print(f)
            with open(f, "r", encoding="utf8") as in_file:
                for line in in_file:
                    out.write(line)

def remove_duplicate_lines(in_file: str, out_file: str):
    visited_hashes = []

    with open(in_file, 'r', encoding="utf8") as input:
        with open(out_file, 'wt', encoding="utf8") as output:

            for line in input.readlines():
                line_hash = hashlib.md5(line.strip().encode("utf8")).hexdigest()

                if line_hash not in visited_hashes:

                    output.write(line)
                    visited_hashes.append(line_hash)

def sort_lines(in_file: str, out_file: str, sep: str='\t', n: int=1):
    # sort lines by first n elements

    with open(in_file, 'r', encoding="utf8") as input:
        with open(out_file, 'wt', encoding="utf8") as output:

            lines = input.readlines()
            lines.sort(key=lambda x: sep.join(x.split(sep)[:n]))

            for line in lines:
                output.write(line)

def convert_to_np_array(string_list: List[str]):
    return np.array([x.split(' ') for x in string_list], dtype=float)

def count_lines(folder: str):
    return sum([file_len(folder + "/" + file) for file in os.listdir(folder)])

def count_lines_all(folder: str):
    for file in os.listdir(folder):
        filepath = os.path.join(folder, file)
        if os.path.isfile(filepath):
            print(filepath, file_len(filepath))

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
    remove_duplicate_lines(tmp_sorted, out_file)

def build_index(filename, sort_col):
    index = []
    f = open(filename, encoding="utf8")
    while True:
        offset = f.tell()
        line = f.readline()
        if not line:
            break
        length = len(line)
        col = line.split(',')[sort_col].strip()
        index.append((col, offset, length))
    f.close()
    index.sort()
    return index

def write_sorted(filename, out_file, col_sort):
    index = build_index(filename, col_sort)
    with open(filename, encoding="utf8") as f:
        with open(out_file, "w", encoding="utf8") as g:

            print("Sorting file %s ..." % filename)

            for col, offset, length in index:
                f.seek(offset)
                g.write(f.read(length))

def file_to_folders(filename, out_dir):
    dir_name = os.path.join(out_dir, filename.split(".")[1])
    os.mkdir(dir_name)

    header = ""
    letter = ''
    g = None

    with open(filename, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            if i == 0:
                header = line
            if line[0] == letter:
                g.write(line)
            else:
                letter = line[0]

                if g:
                    g.close()

                new_file = os.path.join(dir_name, letter)

                if os.path.exists(new_file):
                    g = open(os.path.join(dir_name, letter), "a", encoding="utf8")
                    g.write(line)
                else:
                    g = open(os.path.join(dir_name, letter), "w", encoding="utf8")
                    g.write(header)
                    g.write(line)

    if g:
        g.close()

def get_multisense_words(in_file, out_file, sep='|'):
    wc = count_words(in_file, sep=sep)
    multisense_words = [w for w, c in wc.items() if c > 1]
    filter_file_by_words(in_file, multisense_words, out_file, split_by=sep, split_by_2=sep)

def fix_indices(in_file, out_file, sep="\t", batch_size=100, indices_idx=3, use_form=True):
    lemmatizer = classla.Pipeline('sl', processors='tokenize,pos,lemma', use_gpu=True)

    with open(in_file, "r", encoding="utf8") as f:
        data = [line.split(sep) for line in f]

    print(f"[{get_now_string()}] Data loaded!")

    words = [d[0] for d in data]
    sentences_split = [d[-1].strip().split(" ") for d in data]

    n = len(words) // batch_size
    with open(out_file, "w", encoding="utf8") as f:

        for batch in range(n):
            print(f"[{get_now_string()}] Starting batch {batch} / {n}")

            start = batch * batch_size
            end = min((batch + 1) * batch_size, len(words))

            fix_indices_batch(data[start:end], sentences_split[start:end], lemmatizer, f,
                              sep, indices_idx=indices_idx, use_form=use_form)

        if n == 0:
            fix_indices_batch(data, sentences_split, lemmatizer, f, sep, indices_idx=indices_idx, use_form=use_form)

def fix_indices_batch(data_batch, sentences_split_batch, lemmatizer, out_file, sep, indices_idx=3, use_form=True, insert_form=False):
    data_to_fix = []

    for d, s in zip(data_batch, sentences_split_batch):
        word, idx = d[0], d[indices_idx]
        word = word.replace("-", " - ")
        word = word.replace("  ", " ")
        word_len = word.count(" ") + 1

        # find previously defined word form (may be incorrect)
        if use_form:
            form = d[indices_idx - 2]
        else:
            form = " ".join(s[int(idx): int(idx) + word.count(" ") + 1])

        matches = [get_close_matches(w, s, cutoff=0.1) for w in word.split(" ")]

        # if lengths and first three chars match, it is probably ok
        if form.count(" ") == word.count(" ") and form.lower()[:3] == word.lower()[:3]:
            out_file.write(sep.join(d))

        # if all the words are matched in text and they equate the form, it is ok
        elif all(matches) and " ".join([m[0] for m in matches]) == form:
            if word.count(" ") < form.count(" "):
                d[indices_idx - 2] = " ".join(form.split(" ")[:word.count(" ") + 1])
            out_file.write(sep.join(d))

        # take a look at closest matches for word
        else:
            close = [get_close_matches(w, s, cutoff=0.2) for w in word.split(" ")]

            # if match is close to original index, that could be the right match
            if all(close) and abs(s.index(close[0][0]) - int(idx)) < 5:
                counts = [s.count(m[0]) for m in close]
                indices = [s.index(c[0]) for c in close]

                # does the first match occur only once and are indices consecutive?
                if counts[0] == 1 and indices == list(range(indices[0], indices[-1] + 1)):
                    d[indices_idx] = str(s.index(close[0][0]))

                    if use_form:
                        close_forms = [c[0] for c in close]

                        dists = [abs(s.index(fp) - int(idx)) for fp, word_part in zip(close_forms, word.split(" "))]
                        if all([x < 5 for x in dists]):
                            d[indices_idx - 2] = " ".join(close_forms)

                    out_file.write(sep.join(d))

                else:
                    data_to_fix.append(d)

            # one word
            elif all(close) and word_len == 1:
                d[indices_idx] = str(s.index(close[0][0]))

                if use_form:
                    d[indices_idx - 2] = close[0][0]

                out_file.write(sep.join(d))

            else:
                data_to_fix.append(d)

    if not data_to_fix:
        return

    sentences_split_to_fix = [d[-1].strip().split(" ") for d in data_to_fix]
    words_to_fix = [d[0] for d in data_to_fix]
    indices_to_fix = [int(d[indices_idx]) for d in data_to_fix]

    print(f"[{get_now_string()}] Lemmatizing sentences ...")
    word_lemmas, lemmatized_sentences = get_sentence_lemmas_list(words_to_fix, sentences_split_to_fix, lemmatizer=lemmatizer)

    for ii, lemma, sentence_lemmas, idx in zip(range(len(word_lemmas)), word_lemmas, lemmatized_sentences, indices_to_fix):

        if lemma != lemmatized_sentences[ii][idx] and lemma != sentences_split_to_fix[ii]:
            new_idx, found_lemma = find_new_index(lemma, idx, sentence_lemmas)

            len_lemma = lemma.count(" ") + 1

            indices_to_fix[ii] = new_idx
            data_to_fix[ii][indices_idx] = str(new_idx)

            if use_form:
                data_to_fix[ii][indices_idx - 2] = " ".join(sentences_split_to_fix[ii][new_idx: new_idx + len_lemma])

        out_file.write(sep.join(data_to_fix[ii]))

def find_new_index(lemma, idx, sentence_lemmas):
    n = lemma.count(" ") + 1
    m = len(sentence_lemmas)

    word_form = " ".join(sentence_lemmas[idx: idx + n])
    if lemma.lower() == word_form.lower():
        return idx, word_form

    i = 1
    while idx + i < m or idx - i >= 0:

        new_idx = idx + i
        if new_idx < m:

            word_form = " ".join(sentence_lemmas[new_idx: new_idx + n])
            if lemma.lower() == word_form.lower():
                return new_idx, word_form

        new_idx = idx - i
        if new_idx >= 0:

            word_form = " ".join(sentence_lemmas[new_idx: new_idx + n])
            if lemma.lower() == word_form.lower():
                return new_idx, word_form

        i += 1

    if lemma in sentence_lemmas:
        idx = sentence_lemmas.index(lemma)
        return idx, lemma

    close = get_close_matches(lemma, sentence_lemmas, cutoff=0.4)
    #print(f"{lemma} close matches:", close)
    if close:
        idx = sentence_lemmas.index(close[0])
        return idx, " ".join(sentence_lemmas[idx: idx + n])
    else:
        print("Something was not ok", lemma, sentence_lemmas, idx, sentence_lemmas[idx: idx + n])
        return idx, " ".join(sentence_lemmas[idx: idx + n])

def get_now_string():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def get_pos(pos_label):
    pos_dict = {"S": "samostalnik",
                "G": "glagol",
                "P": "pridevnik",
                "R": "prislov",
                "Z": "zaimek",
                "K": "števnik",
                "D": "predlog",
                "V": "veznik",
                "L": "členek",
                "M": "medmet",
                "O": "okrajšava"}
    pos = pos_label[0]
    if pos in pos_dict.keys():
        return pos_dict[pos]
    else:
        return "N/A"

def get_pos_short(pos_label):
    pos_dict = {"samostalnik": "S",
                "glagol": "G",
                "pridevnik": "P",
                "prislov": "R",
                "zaimek": "Z",
                "števnik": "K",
                "predlog": "D",
                "veznik": "V",
                "členek": "L",
                "medmet": "M",
                "okrajšava": "O",
                "N/A": "U"}
    pos = pos_label[0]
    if pos in pos_dict.keys():
        return pos_dict[pos]
    else:
        return "U"

