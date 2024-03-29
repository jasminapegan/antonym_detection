import json
import os
import re
from io import TextIOWrapper
from lxml import etree as ET
from multiprocessing.pool import ThreadPool as Pool
from typing import Pattern, List, Iterable

import file_helpers
from file_helpers import get_unique_words_pos
from data.lemmatization import get_word_lemmas_list


def get_sentences_multiprocess(gigafida_dir: str, words_file: str, tmp_dir: str="tmp", sample_size: int=100,
                               folders_range: List[int]=list(range(100)), sep="|"):
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
    print("[%s] Starting ..." % file_helpers.get_now_string())

    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    preexisting = deserialize_globals()
    if not preexisting:

        prepare_word_data(words_file, sep=sep)
        print("[%s] Word data prepared!" % file_helpers.get_now_string())
        serialize_globals()

    pool = Pool(min(20, len(folders_range)))

    for i in folders_range:
        pool.apply_async(get_sentences_part, (gigafida_dir, tmp_dir, i, sample_size),
                         error_callback=lambda x: print("Thread Error:", x))

    pool.close()
    pool.join()

    serialize_globals()
    print("[%s] Finished." % file_helpers.get_now_string())

def serialize_globals(tmp_folder="tmp"):
    global words_data

    words_data_copy = {}
    for word in words_data.keys():
        words_data_copy[word] = {
            'lemma': words_data[word]['lemma'],
            'pos': words_data[word]['pos']
        }

    tmp_file = os.path.join(tmp_folder, "globals.txt")

    with open(tmp_file, "w", encoding="utf8") as f:
        f.write(json.dumps(words_data_copy))

    print("[%s] Globals serialized." % file_helpers.get_now_string())

def deserialize_globals(tmp_folder="tmp"):
    global words_data

    tmp_file = os.path.join(tmp_folder, "globals.txt")

    if not os.path.exists(tmp_file):
        words_data = {}
        return False

    else:
        with open(tmp_file, "r", encoding="utf8") as f:
            lines = f.readlines()
            words_data = json.loads(lines[0])

        for word in words_data.keys():
            words_data[word]['pattern'] = re.compile(r'\b%s\b' % words_data[word]['lemma'])

        print("[%s] Globals deserialized." % file_helpers.get_now_string())
        return True

def finalize_sentence_search(words_file: str, sample_out: str, info_out: str, tmp_dir: str="tmp", sample_size: int=100,
                             folders_range: Iterable[int]=range(100), sep: str="\t"):
    files = [os.path.join(tmp_dir, "%02d.txt" % i) for i in folders_range]
    all_sentences_file = gather_sentence_data(files, tmp_dir=tmp_dir)
    get_sample_sentences(words_file, all_sentences_file, sample_out, info_out, sample_size=sample_size)

    print("[%s] Missing words\n" % file_helpers.get_now_string(), missing_words(words_file, sample_out, sep=sep))

def prepare_word_data(words_file: str, sep: str="|"):
    global words_data

    words = get_unique_words_pos(words_file, sep=sep)
    phrases = {w: pos for w, pos in words.items() if re.findall("[-\s]", w)}
    singular_words = {w: pos for w, pos in words.items() if w not in phrases}

    pattern = re.compile("-|\s+")
    deconstructed_words = [pattern.split(w) for w in phrases.keys()]

    words_lemmatized = get_word_lemmas_list([" ".join(w) for w in deconstructed_words])
    words_lemmatized += singular_words

    patterns = [re.compile(r'\b%s\b' % w) for w in words_lemmatized]

    words = dict(phrases)
    words.update(singular_words)

    words_data = {
        word: {'lemma': l,
               'pattern': p,
               'pos': {x: 0 for x in words[word]}}
        for word, l, p in zip(words.keys(), words_lemmatized, patterns)
    }

    serialize_globals()


def get_sentences_part(gigafida_dir: str, out_path: str, i: int, limit: int):
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
        global words_data

        with open(os.path.join(out_path, "%02d.txt" % i), "w", encoding="utf8") as outf:
            print("[%s] Opened out-file #%d" % (file_helpers.get_now_string(), i))

            gigafida_subdir = os.path.join(gigafida_dir, "GF%02d/" % i)
            re_whitespace = re.compile(r"\s+")

            for j, file in enumerate(os.listdir(gigafida_subdir)):
                if j % 100 == 0:
                    print("[%s] Opened file #%d / 384 in folder #%d" % (file_helpers.get_now_string(), j, i))

                tree = ET.parse(os.path.join(gigafida_subdir, file))

                if not words_data:
                    return

                get_sentences_from_tree(tree, outf, re_whitespace, limit=limit)

    except Exception as e:
        print("[%s] Get Sentences Error:" % file_helpers.get_now_string(), e)

    finally:
        print("[%s] Closed out-file #%d" % (file_helpers.get_now_string(), i))

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

    global words_data

    root = tree.getroot()
    for p in root[1][1]:

        if not words_data:
            return

        sentences, lemma_sentences, pos_tags_list = parse_paragraph(p, re_whitespace)
        process_sentences(sentences, lemma_sentences, pos_tags_list, min_tokens, limit, out_file)

def process_sentences(sentences, lemma_sentences, pos_tags_list, min_tokens, limit, out_file):
    global words_data

    n = len(sentences)

    for i in range(n):
        sentence = sentences[i]
        lemma_sentence = lemma_sentences[i]
        pos_tags = pos_tags_list[i]

        for word, word_data in {**words_data}.items(): # no problem if we get too much of some words
            lemma, pattern, pos_count = word_data['lemma'], word_data['pattern'], word_data['pos']

            match = pattern.search(lemma_sentence)

            if match:
                idx = match.start()

                sentence, idx, pos_tags = lengthen_sentence(sentence, sentences, lemma_sentences, pos_tags,
                                                            pos_tags_list, i, min_tokens, idx)

                if sentence.count(' ') < min_tokens:
                    continue

                spaces = lemma_sentence[:idx].count(' ')
                pos_tag = pos_tags[spaces]
                pos = file_helpers.get_pos(pos_tags[spaces][0])

                if pos == "N/A" or pos in words_data[word]['pos'].keys():

                    out_file.write("\t".join([word, pos_tag, str(spaces), sentence]) + "\n")
                    update_word_count_in_globals(word, pos_tag, limit)

def lengthen_sentence(sentence: str, sentences: List[str], lemma_sentences: List[str], pos_tags: List[str],
                      pos_tags_list: List[List[str]], i: int, min_tokens: int,
                      idx: int) -> (str, int):
    n = len(sentences)
    k = i

    while sentence.count(' ') < min_tokens and k != 0 and k != n - 1:

        if k + 1 < n:
            sentence += " " + sentences[k + 1]
            pos_tags += ["X"] + pos_tags_list[k + 1]

        elif k > 0:
            sentence = sentences[k - 1] + " " + sentence
            pos_tags = pos_tags_list[k + 1] + ["X"] + pos_tags
            idx += len(lemma_sentences[k - 1]) + 1

    return sentence, idx, pos_tags

def update_word_count_in_globals(word: str, pos_tag: str, limit):
    global words_data

    try:
        if list(words_data[word]['pos'].keys()) == ["N/A"]:
            tag = "N"
            words_data[word]['pos'][0] += 1

            if words_data[word]['pos']['N/A'] >= limit:
                del words_data[word]

                n = len(words_data)

                if n % 100 == 0:
                    print("[%s] Finished sampling word '%s'. Remaining %d words." % (file_helpers.get_now_string(), word, n))

                if n == 0:
                    return
        else:
            tag = file_helpers.get_pos(pos_tag)
            words_data[word]['pos'][tag] += 1

        if words_data[word]['pos'][tag] >= limit:
            del words_data[word]['pos'][tag]

            if not words_data[word]['pos']:
                del words_data[word]

            n = len(words_data)

            if n % 100 == 0:
                print("[%s] Finished sampling word '%s'. Remaining %d words." % (file_helpers.get_now_string(), word, n))

            if n == 0:
                return

    except (ValueError, KeyError) as e:
        pass #print("Failed removing word '%s' from globals: %s" % (word, e))


def parse_paragraph(paragraph: ET.Element, re_whitespace: Pattern) -> (List[str], List[str]):
    """
    Loops through the 'paragraph' element. Returns a list of sentences and a list of corresponding lemmatized sentences.

    :param paragraph: ETree element that represents GigaFida paragraph
    :param re_whitespace: whitespace-representing pattern to replace with ' '
    :return: a pair containing a list of parsed sentences and a list containing lemmatized sentences
    """

    sentences = []
    lemma_sentences = []
    pos_tags_list = []

    for s in paragraph:
        sentence, lemmas, pos_tags = parse_sentence(s, re_whitespace)

        if len(sentence) > 0:
            sentences.append(sentence)
            lemma_sentences.append(lemmas)
            pos_tags_list.append(pos_tags)

    return sentences, lemma_sentences, pos_tags_list

def parse_sentence(s: ET.Element, re_whitespace: Pattern):
    sentence = ""
    pos_tags = []
    lemmas = ""

    for w in s:

        pos_tag = get_pos_tag_from_xml(w)

        if w.tag[-1] == 'w':
            sentence += w.text
            lemmas += w.attrib["lemma"]
            pos_tags.append(pos_tag)

        elif w.tag[-1] == 'S':
            sentence += " "
            lemmas += " "
            pos_tags.append(pos_tag)

        elif w.tag[-1] == 'c':
            sentence += " " + w.text + " "
            lemmas += " " + w.text + " "
            pos_tags.append(pos_tag)

    sentence = re_whitespace.sub(" ", sentence.strip())
    lemmas = re_whitespace.sub(" ", lemmas.strip())

    return sentence, lemmas, pos_tags

def get_pos_tag_from_xml(w):
    if "ana" in w.attrib.keys():
        tag_text = w.attrib["ana"]

        if ":" in tag_text:
            return tag_text.split(":")[1]
        else:
            print("ana tag not complete: %s" % tag_text)

    #else:
    #    print("not found pos:", w.attrib.keys())

    return "X"

def missing_words(words_file: str, data_file: str, sep: str="|") -> List[str]:
    """
    Loops through tsv file 'data_file' contents and counts lines beginning with each word in 'words_file'. Returns a
    list of words that did not occur.

    :param words_file: '|'-separated file that contains words at index 0
    :param data_file: tsv file to search for word occurences
    :return: a list of words that did not occur in 'data_file' at index 0
    """

    #words = file_helpers.get_unique_words(words_file, sep=sep)
    words = get_unique_words_pos(words_file, sep=sep)

    with open(data_file, "r", encoding="utf8") as f:

        for line in f:
            word, pos = line.split("\t")[:2]

            if word in words.keys() and pos in words[word]:
                words[word].remove(pos)
                #words.remove(word)

                if not words[word]:
                    del words[word]

            if not words:
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

    file_helpers.concatenate_files(list(files), all_sentences)
    file_helpers.sort_lines(all_sentences, all_sorted, n=2)
    file_helpers.remove_duplicate_lines(all_sorted, all_deduplicated)

    return all_deduplicated

def get_sample_sentences(words_file: str, sentences_all: str, out_file: str, out_info: str, sample_size: int=100,
                         sep: str="|"):
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

    #words = file_helpers.get_unique_words(words_file, sep="\t")
    words = get_unique_words_pos(words_file, sep=sep)
    words_dict = {w: {p: 0 for p in set(pos + ["N/A"])} for w, pos in words.items()}

    with open(out_file, "w", encoding="utf8") as outf:
        with open(sentences_all, "r", encoding="utf8") as f:
            for line in f:
                w, pos = line.split("\t")[:2]
                pos = file_helpers.get_pos(pos)

                if w not in words_dict.keys():
                    continue # print("Not in keys: %s" % w)

                if words_dict[w][pos] < sample_size:
                    outf.write(line)
                    words_dict[w][pos] += 1

    keyval = list(words_dict.items())
    keyval.sort(key=lambda x: sum(list(x[1].values())))

    with open(out_info, "w", encoding="utf8") as info_file:
        for key, vals in keyval:
            for pos, val in vals.items():
                info_file.write("%s\t%s\t%d\n" % (key, pos, val))


def get_sentence_by_id(file, sentence_id):
    re_whitespace = re.compile(r"\s+")
    regex = re.compile(r'<s xml:id="%s">[.\s]' % sentence_id)
    regex_end = re.compile(r'</s>')
    tree_data = ""
    found = False

    with open(file, "r", encoding="utf8") as f:
        for line in f:

            if found:
                tree_data += line

                if regex_end.match(line):
                    tree = ET.ElementTree(ET.fromstring(tree_data))
                    sentence, lemmas, _ = parse_sentence(tree.getroot(), re_whitespace)
                    return sentence, lemmas

            elif regex.match(line):
                tree_data += line
                found = True

    tree = ET.ElementTree(ET.fromstring(tree_data))
    sentence, lemmas, _ = parse_sentence(tree.getroot(), re_whitespace)
    return sentence, lemmas

