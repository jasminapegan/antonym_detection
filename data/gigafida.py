import os
import re
import xml.etree.ElementTree as ET
from multiprocessing.pool import ThreadPool as Pool

import file_helpers
from file_helpers import get_unique_words
from data.lemmatization import get_word_lemmas_list


def get_sentences_from_gigafida_multiprocess(gigafida_dir, words_file, sample_out, info_out, lemmatize=False, tmp_dir="tmp/", sample_size=100):
    global words
    global words_lemmatized
    global words_count

    words = get_unique_words(words_file)
    words_lemmatized = None

    if lemmatize:
        words_single = [w for w in words if len(w) == 1]
        words_lemmatized = words_single + get_word_lemmas_list([w for w in words if len(w) > 1])

    words_count = {word: 0 for word in words}

    pool = Pool(20)

    for i in range(100):
        print("%d / 100" % i)
        pool.apply_async(get_sentences_from_gigafida_part, (gigafida_dir, tmp_dir + "GF", i, sample_size), error_callback=lambda x: print(x))

    pool.close()
    pool.join()

    print("Sentence samples found, sorting and deduplicating ...")

    get_sample_sentences_from_results([tmp_dir + "GF%02d.txt" % i for i in range(100)], words_file, sample_out, info_out, tmp_dir=tmp_dir, sample_size=sample_size)


def get_sentences_from_gigafida_part(gigafida_dir, out_file, i, limit):
    try:
        global words
        global words_count
        global words_lemmatized

        with open(out_file + "%02d.txt" % i, "w", encoding="utf8") as outf:
            gigafida_subdir = gigafida_dir + "GF%02d/" % i
            re_whitespace = re.compile(r"\s+")

            for j, file in enumerate(os.listdir(gigafida_subdir)):
                tree = ET.parse(gigafida_subdir + file)

                if len(words) == 0:
                    return

                if words_lemmatized:
                    get_sentences_from_tree_multiword_globals(tree, outf, re_whitespace, limit=limit)
                else:
                    get_sentences_from_tree(tree, words, outf, re_whitespace)
    except (NameError, KeyError) as e:
        print(e)
        return
    except Exception as e2:
        print(e2)
        return


def get_sentences_from_gigafida(gigafida_dir, words_file, out_file, lemmatize=False):
    re_whitespace = re.compile(r"\s+")
    words = get_unique_words(words_file)
    words_single = [w for w in words if len(w) == 1]
    words_lemmatized = words_single + get_word_lemmas_list([w for w in words if len(w) > 1])

    with open(out_file, "w", encoding="utf8") as out_file:

        for i in range(100):
            print("%d / 100" % i)

            gigafida_subdir = gigafida_dir + "GF%02d/" % i
            for j, file in enumerate(os.listdir(gigafida_subdir)):

                tree = ET.parse(gigafida_subdir + file)
                #get_sentences_from_tree(tree, words, out_file, re_whitespace)
                get_sentences_from_tree_multiword(tree, words, words_lemmatized, out_file, re_whitespace)


def get_sentences_from_tree(tree, words, out_file, re_whitespace, missing=False):
    root = tree.getroot()

    for p in root[1][1]:
        for s in p:

            sentence = ""
            words_found = []

            for w in s:

                if w.tag[-1] == 'w':
                    sentence += w.text

                    if not missing:
                        lemma = w.attrib["lemma"]

                        if lemma in words:
                            words_found.append([lemma, w.text])

                elif w.tag[-1] == 'S':
                    sentence += " "

                elif w.tag[-1] == 'c':
                    sentence += " " + w.text + " "

            sentence = re_whitespace.sub(" ", sentence).strip()
            spaces = sentence.count(" ")

            if missing:
                words_found = [(w, w) for w in words if " " + w + " " in sentence]

            if len(words_found) != 0 and spaces > 8:

                sentence = sentence.replace('\t', ' ')
                sentence_split = sentence.split(" ")

                for w, wo in words_found:
                    if " " in wo:
                        idx = sentence_split.index(wo.split(" ")[0])
                    else:
                        idx = sentence_split.index(wo)
                    out_file.write("\t".join([w, str(idx), sentence]) + "\n")


def get_sentences_from_tree_multiword_globals(tree, out_file, re_whitespace, limit=100, min_words=8):
    root = tree.getroot()
    global words
    global words_count
    global words_lemmatized

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
                    while sentence.count(' ') < min_words and k != 0 and k != n - 1:
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


def get_sentences_from_tree_multiword(tree, words, words_lemmatized, out_file, re_whitespace, limit=100, min_words=8):
    root = tree.getroot()
    words_count = {word: 0 for word in words}

    for p in root[1][1]:
        sentences, lemma_sentences = parse_paragraph(p, re_whitespace)
        n = len(sentences)

        for i, (sentence, lemma_sentence) in enumerate(zip(sentences, lemma_sentences)):
            for word, word_lemmatized in zip(words[::-1], words_lemmatized[::-1]):

                if len(words) == 0:
                    return

                j = lemma_sentence.find(word_lemmatized)
                if j != -1:

                    k = i
                    while sentence.count(' ') < min_words and k != 0 and k != n - 1:
                        if k + 1 < n:
                            sentence += " " + sentences[k+1]
                        elif k > 0:
                            sentence = sentences[k-1] + " " + sentence

                    idx = lemma_sentence[:j].count(' ')
                    out_file.write("\t".join([word, str(idx), sentence]) + "\n")

                    words_count[word] += 1
                    if words_count[word] >= limit:

                        del words_count[word]
                        words.remove(word)
                        words_lemmatized.remove(word_lemmatized)


def parse_paragraph(paragraph, re_whitespace):
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


def missing_words(words_file, data_location, is_dir=True):
    words = get_unique_words(words_file)

    if is_dir:
        for i in range(100):
            with open(data_location + "sentences_%02d.txt" % i, "r", encoding="utf8") as f:

                for line in f:
                    word = line.split("\t")[0]

                    if word in words:
                        words.remove(word)

                    if len(words) == 0:
                        return words

    else:
        with open(data_location, "r", encoding="utf8") as f:

            for line in f:
                word = line.split("\t")[0]

                if word in words:
                    words.remove(word)

                if len(words) == 0:
                    return words

    return words


def get_missing_sentences_from_gigafida(gigafida_dir, missing_words_file, out_file):
    re_whitespace = re.compile(r"\s+")
    with open(missing_words_file, "r", encoding="utf8") as f:
        words = [word.strip() for word in f]

    with open(out_file, "w", encoding="utf8") as out_file:

        for i in range(100):
            print("%d / 100" % i)

            gigafida_subdir = gigafida_dir + "GF%02d/" % i
            for j, file in enumerate(os.listdir(gigafida_subdir)):

                tree = ET.parse(gigafida_subdir + file)
                get_sentences_from_tree(tree, words, out_file, re_whitespace, missing=True)


def get_sample_sentences_from_results(gigafida_files, words_file, out_file, out_info, tmp_dir="tmp/", sample_size=100):
    all_sentences = tmp_dir + "all_sentences.txt"
    all_sorted = tmp_dir + "all_sentences_sorted.txt"
    all_deduplicated = tmp_dir + "all_sentences_deduplicated.txt"

    file_helpers.concatenate_files(gigafida_files, all_sentences)
    file_helpers.sort_lines(all_sentences, all_sorted)
    file_helpers.remove_duplicate_lines(all_sorted, all_deduplicated, range=1)

    get_sample_sentences(words_file, all_deduplicated, out_file, out_info, sample_size=sample_size)


def get_sample_sentences(words_file, sentences_all, out_file, out_info, sample_size=100):
    words = get_unique_words(words_file)
    words_dict = {w: 0 for w in words}

    with open(out_file, "w", encoding="utf8") as outf:
        with open(sentences_all, "r", encoding="utf8") as f:
            for line in f:
                w, _, _ = line.split("\t")

                if w not in words_dict.keys():
                    print("Not in keys: %s" % w)
                    """w1, w2, w3 = w.replace(' ', '-'), w.replace(' - ', '-'), w + " se"
                    if w1 in words_dict.keys():
                        w = w1
                    elif w2 in words_dict.keys():
                        w = w2
                    else:
                        w = w3"""

                if words_dict[w] < sample_size:
                    outf.write(line)
                    words_dict[w] += 1

    keyval = list(words_dict.items())
    keyval.sort(key=lambda x: x[1])

    with open(out_info, "w", encoding="utf8") as info_file:
        for key, val in keyval:
            info_file.write("%s\t%d\n" %(key, val))

