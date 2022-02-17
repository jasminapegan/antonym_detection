import os
import re
import xml.etree.ElementTree as ET
from multiprocessing.pool import ThreadPool as Pool

import file_helpers
from file_helpers import get_unique_words
from data.lemmatization import get_word_lemmas_list


def get_sentences_from_gigafida_multiprocess(gigafida_dir, words_file, sample_out, info_out, lemmatize=False, tmp_dir="tmp", sample_size=100, n_folders=100):
    global words
    global words_lemmatized
    global words_count

    words = get_unique_words(words_file)
    words_lemmatized = None

    if lemmatize:
        words_single = [w for w in words if len(w) == 1]
        words_lemmatized = words_single + get_word_lemmas_list([w for w in words if len(w) > 1])
    else:
        words_lemmatized = words[:]

    words_count = {word: 0 for word in words}

    pool = Pool(min(20, n_folders))

    for i in range(n_folders):
        print("%d / %d" % (i, n_folders))
        pool.apply_async(get_sentences_from_gigafida_part, (gigafida_dir, tmp_dir, i, sample_size), error_callback=lambda x: print(x))

    pool.close()
    pool.join()

    print("Sentence samples found, sorting and deduplicating ...")

    files = [os.path.join(tmp_dir, "%02d.txt" % i) for i in range(n_folders)]
    get_sample_sentences_from_results(files, words_file, sample_out, info_out, tmp_dir=tmp_dir, sample_size=sample_size)

    print("Missing words\n", missing_words(words_file, sample_out, is_dir=False))


def get_sentences_from_gigafida_part(gigafida_dir, out_path, i, limit):
    try:
        global words
        global words_count
        global words_lemmatized

        with open(os.path.join(out_path, "%02d.txt" % i), "w", encoding="utf8") as outf:
            gigafida_subdir = os.path.join(gigafida_dir, "GF%02d/" % i)
            re_whitespace = re.compile(r"\s+")

            for j, file in enumerate(os.listdir(gigafida_subdir)):
                tree = ET.parse(gigafida_subdir + file)

                if len(words) == 0:
                    return

                get_sentences_from_tree(tree, outf, re_whitespace, limit=limit)

    except (NameError, KeyError) as e:
        print(e)
        return
    except Exception as e2:
        print(e2)
        return


def get_sentences_from_tree(tree, out_file, re_whitespace, limit=100, min_words=8):
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


def missing_words(words_file, data_location, is_dir=True, n_folders=100):
    words = get_unique_words(words_file)

    if is_dir:
        for i in range(n_folders):

            file = os.path.join(data_location, "%02d.txt" % i)
            with open(file, "r", encoding="utf8") as f:

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


def get_sample_sentences_from_results(gigafida_files, words_file, out_file, out_info, tmp_dir="tmp", sample_size=100):
    all_sentences = os.path.join(tmp_dir, "sentences.txt")
    all_sorted = os.path.join(tmp_dir, "sentences_sorted.txt")
    all_deduplicated = os.path.join(tmp_dir, "sentences_deduplicated.txt")

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

                if words_dict[w] < sample_size:
                    outf.write(line)
                    words_dict[w] += 1

    keyval = list(words_dict.items())
    keyval.sort(key=lambda x: x[1])

    with open(out_info, "w", encoding="utf8") as info_file:
        for key, val in keyval:
            info_file.write("%s\t%d\n" % (key, val))
