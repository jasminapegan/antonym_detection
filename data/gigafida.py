import os
import re
from io import TextIOWrapper
import xml.etree.ElementTree as ET
from multiprocessing.pool import ThreadPool as Pool
from typing import Pattern, List, Iterable

import file_helpers
from file_helpers import get_unique_words
from data.lemmatization import get_word_lemmas_list


def get_sentences_from_gigafida_multiprocess(gigafida_dir: str, words_file: str, sample_out: str, info_out: str,
                                             tmp_dir: str="tmp", sample_size: int=100, n_folders: int=100):
    """
    Searches for word usage samples in GigaFida files.

    Loops through files in 'gigafida_dir' and gets 'sample_size' examples of word usages from 'words_file'. Outputs
    sentences data (tsv file with word, token index, sentence data) in 'sample_out' and information about number of
    word occurences in tsv file 'info_out'. Intermediate files are stored in 'tmp_dir' folder. At the end, looks for
    missing words in 'sample_out' and prints them out.

    :param gigafida_dir: folder containing GigaFida files
    :param words_file: '|'-separated file containing word data at index 0
    :param sample_out: file to output sample sentences
    :param info_out: file to output info on number of word occurences in 'sample_out'
    :param tmp_dir: temporary directory for storing intermediate files (default 'tmp')
    :param sample_size: number of sentences to find for each word (default 100)
    :param n_folders: number of folders to loop through in GigaFida directory (default 100, which is all)
    :return: None
    """

    global words
    global words_lemmatized
    global words_count

    # create a list of words + lemmatized multiwords
    words = get_unique_words(words_file)
    words_lemmatized = get_word_lemmas_list([w for w in words if len(w) > 1])
    words_lemmatized += [w for w in words if len(w) == 1]
    words_count = {word: 0 for word in words}

    pool = Pool(min(20, n_folders))
    for i in range(n_folders):

        print("%d / %d" % (i, n_folders))
        pool.apply_async(get_sentences_from_gigafida_part, (gigafida_dir, tmp_dir, i, sample_size), error_callback=lambda x: print(x))

    pool.close()
    pool.join()

    print("Sentence samples found, sorting and deduplicating ...")

    files = [os.path.join(tmp_dir, "%02d.txt" % i) for i in range(n_folders)]
    all_sentences_file = gather_sentence_data(files, tmp_dir=tmp_dir)
    get_sample_sentences(words_file, all_sentences_file, sample_out, info_out, sample_size=sample_size)

    print("Missing words\n", missing_words(words_file, sample_out))

def get_sentences_from_gigafida_part(gigafida_dir: str, out_path: str, i: int, limit: int):
    """
    Searches 'i'-th file in 'gigafida_dir' for words (which are stored in global variable 'words') and stores found data
    in a new file in the 'out_path' directory. GigaFida directory contains subdirectories such as 'GF00', therefore
    knowing index of the subdirectory is enough to access it.

    :param gigafida_dir: GigaFida directory
    :param out_path: folder in which output files are written
    :param i: index of the subdirectory to be searched
    :param limit: how many samples of each word are needed
    :return: None
    """
    try:
        global words
        global words_count
        global words_lemmatized

        with open(os.path.join(out_path, "%02d.txt" % i), "w", encoding="utf8") as outf:

            gigafida_subdir = os.path.join(gigafida_dir, "GF%02d/" % i)
            re_whitespace = re.compile(r"\s+")

            for j, file in enumerate(os.listdir(gigafida_subdir)):
                tree = ET.parse(os.path.join(gigafida_subdir + file))

                if len(words) == 0:
                    return

                get_sentences_from_tree(tree, outf, re_whitespace, limit=limit)

    except (NameError, KeyError) as e:
        print(e)
        return
    except Exception as e2:
        raise e2

def get_sentences_from_tree(tree: ET.ElementTree, out_file: TextIOWrapper, re_whitespace: Pattern, limit: int, min_tokens: int=8):
    """
    Loops through the 'tree' and searches for words stored in global variable. Stores sentences containing these words
    into pre-made 'out_file' file wrapper for speed.

    :param tree: ETree containing data of a GigaFida file
    :param out_file: file wrapper to write data to output file
    :param re_whitespace: pre-compiled pattern to match whitespace groups (for speed)
    :param limit: how many word occurences are needed
    :param min_tokens: minimal number of tokens for the sentence to be included (detault 8)
    :return: None
    """

    global words
    global words_count
    global words_lemmatized

    root = tree.getroot()
    for p in root[1][1]:

        if len(words) == 0:
            return

        sentences, lemma_sentences = parse_paragraph(p, re_whitespace)
        n = len(sentences)

        for i, (sentence, lemma_sentence) in enumerate(zip(sentences, lemma_sentences)):
            for word, word_lemmatized in zip(words[::-1], words_lemmatized[::-1]):

                j = lemma_sentence.find(word_lemmatized)
                if j != -1:

                    k = i
                    while sentence.count(' ') < min_tokens and k != 0 and k != n - 1:
                        if k + 1 < n:
                            sentence += " " + sentences[k+1]
                        elif k > 0:
                            sentence = sentences[k-1] + " " + sentence

                    idx = sentence[:j].count(' ')
                    out_file.write("\t".join([word, str(idx), sentence]) + "\n")

                    words_count[word] += 1
                    if words_count[word] >= limit:

                        del words_count[word]
                        words.remove(word)
                        words_lemmatized.remove(word_lemmatized)

                        if len(words) == 0:
                            return

def parse_paragraph(paragraph: ET.Element, re_whitespace: Pattern) -> (List[str], List[str]):
    """
    Loops through the 'paragraph' element. Returns a list of sentences and a list of corresponding lemmatized sentences.

    :param paragraph: ETree element that represents GigaFida paragraph
    :param re_whitespace: whitespace-representing pattern to replace with ' '
    :return: a pair containing a list of parsed sentences and a list containing lemmatized sentences
    """

    sentences = []
    lemma_sentences = []

    for s in paragraph:
        sentence = ""
        lemmas = ""

        for w in s:

            if w.tag[-1] == 'w':
                sentence += w.text
                lemmas += w.attrib["lemma"]

            elif w.tag[-1] == 'S':
                sentence += " "
                lemmas += " "

            elif w.tag[-1] == 'c':
                sentence += " " + w.text + " "
                lemmas += " " + w.text + " "

        sentence = re_whitespace.sub(" ", sentence).strip()
        lemmas = re_whitespace.sub(" ", lemmas).strip()

        sentences.append(sentence)
        lemma_sentences.append(lemmas)

    return sentences, lemma_sentences

def missing_words(words_file: int, data_file: str) -> List[str]:
    """
    Loops through tsv file 'data_file' contents and counts lines beginning with each word in 'words_file'. Returns a
    list of words that did not occur.

    :param words_file: '|'-separated file that contains words at index 0
    :param data_file: tsv file to search for word occurences
    :return: a list of words that did not occur in 'data_file' at index 0
    """

    words = get_unique_words(words_file)

    with open(data_file, "r", encoding="utf8") as f:

        for line in f:
            word = line.split("\t")[0]

            if word in words:
                words.remove(word)

            if len(words) == 0:
                return words

    return words

def gather_sentence_data(files: Iterable[str], tmp_dir: str="tmp"):
    """
    Gathers, sorts and deduplicates files from 'files' collection. Returns final file location in temporary directory.

    :param files: list of files with gathered results from GigaFida
    :param tmp_dir: list of files with gathered results from GigaFida
    :return: location of edited file in temporary directory
    """

    all_sentences = os.path.join(tmp_dir, "sentences.txt")
    all_sorted = os.path.join(tmp_dir, "sentences_sorted.txt")
    all_deduplicated = os.path.join(tmp_dir, "sentences_deduplicated.txt")

    file_helpers.concatenate_files(files, all_sentences)
    file_helpers.sort_lines(all_sentences, all_sorted)
    file_helpers.remove_duplicate_lines(all_sorted, all_deduplicated, range=1)

    return all_deduplicated

def get_sample_sentences(words_file: str, sentences_all: str, out_file: str, out_info: str, sample_size: int=100):
    """
    Reads a sample of sentences gathered from GigaFida. Outputs sentences data to 'out_file' and info about number of
    sentences per word in 'out_info' file.

    :param words_file: words of interest tsv data file with word at index 0
    :param sentences_all: tsv file with sorted words and sentences data
    :param out_file: output tsv file; contains word, index of first token in sentence, sentence
    :param out_info: output tsv file; contains word and count of sentences containing this word
    :param sample_size: max number of sentences per word
    :return: None
    """

    words = get_unique_words(words_file)
    words_dict = {w: 0 for w in words}

    with open(out_file, "w", encoding="utf8") as outf:
        with open(sentences_all, "r", encoding="utf8") as f:
            for line in f:
                w, _, _ = line.split("\t")

                if w not in words_dict.keys():
                    print("Not in keys: %s" % w)

                if words_dict[w] < sample_size:
                    outf.write(line)
                    words_dict[w] += 1

    keyval = list(words_dict.items())
    keyval.sort(key=lambda x: x[1])

    with open(out_info, "w", encoding="utf8") as info_file:
        for key, val in keyval:
            info_file.write("%s\t%d\n" % (key, val))
