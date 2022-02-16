import os
import pickle
from random import shuffle
import numpy as np


def get_words_data_from_file(words_file):
    words = []

    with open(words_file, "r", encoding="utf8") as in_file:
        for i, line in enumerate(in_file):

            # skip header
            if i != 0:
                word, word_type, num_of_meaning, description = line.strip().split('|')
                words.append({"word": word,
                              "type": word_type,
                              "num": num_of_meaning,
                              "description": description})
    return words


def words_data_to_dict(words_file):
    words = get_words_data_from_file(words_file)
    words_dict = {}

    for word_data in words:
        word = word_data["word"]
        if word not in words_dict.keys():
            words_dict[word] = [word_data]
        else:
            words_dict[word].append(word_data)
    return words_dict

def save_json_word_data(words_file, out_file):
    words_dict = words_data_to_dict(words_file)

    with open(out_file, "wb") as outjson:
        pickle.dump(words_dict, outjson)


def save_json_word_data_from_multiple(words_file1, words_file2, out_file):
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


def load_json_word_data(json_file):
    with open(json_file, "rb") as words_dict_json:
        return pickle.load(words_dict_json)


def get_unique_words(word_file, sep='|'):
    return list(set([line[0] for line in load_file(word_file, sep=sep)]))


def load_file(file, limit=None, sep='\t'):
    data = []
    with open(file, "r", encoding="utf8") as f:

        for i, line in enumerate(f):
            data.append(line.strip().split(sep))

            if i == limit:
                return data

    return data


def file_len(filename):
    i = 0
    with open(filename, encoding='utf8') as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def count_lines(folder):
    return sum([file_len(folder + "/" + file) for file in os.listdir(folder)])


def load_sentences_embeddings_file(file, limit=None):
    data = []
    with open(file, "r", encoding="utf8") as f:

        for i, line in enumerate(f):
            data.append(line.strip().split("\t"))

            if i == limit:
                break

    words = [d[0] for d in data]
    word_forms = [d[1] for d in data]
    sentences = [d[2] for d in data]
    embeddings = convert_to_np_array([d[3] for d in data])

    return words, word_forms, sentences, embeddings


def load_grouped_data(file, start=0, only_word=None):
    data = []
    idx = start
    found = False
    word = None

    max_idx = file_len(file)

    with open(file, "r", encoding="utf8") as f:
        for i, line in enumerate(f):

            if i < idx:
                continue

            data_line = line.strip().split("\t")
            data_word = data_line[0]

            if word is None or data_word == word:
                data_line[0] = data_word
                data.append(data_line)
                word = data_word
                found = True

            if found and data_word != word:
                idx = i

                if only_word:
                    return data, idx

    return data, max_idx


def load_sentences_embeddings_file_grouped(file, start=0, v2=True, only_word=True):
    data, idx = load_grouped_data(file, start=start, only_word=only_word)

    words = [d[0] for d in data]
    sentences = [d[-2] for d in data]
    embeddings = convert_to_np_array([d[-1] for d in data])

    if v2:
        return words, sentences, embeddings, idx
    else:
        word_forms = [d[1] for d in data]
        return words, word_forms, sentences, embeddings, idx


def load_validation_file_grouped(file, only_word=None, start=0, all_strings=False, embeddings=True, indices=False, sentence_idx=2):
    if only_word:
        data, idx = load_grouped_data(file, start=start, only_word=only_word)
    else:
        data = load_file(file, sep='\t')
    words_data = {}

    for line in data:

        word, label, sentence = line[0], line[1], line[sentence_idx]
        embedding = None
        index = None

        if embeddings:
            embedding = line[-1]
            if not all_strings:
                embedding = convert_to_np_array(line[-1])

        if indices:
            index = line[-2]
            if not all_strings:
                index = int(line[-2])

        if word in words_data.keys():
            if embeddings:
                words_data[word]['embeddings'].append(embedding)
            if indices:
                words_data[word]['indices'].append(index)
            words_data[word]['labels'].append(label)
            words_data[word]['sentences'].append(sentence)

        else:
            words_data[word] = {'labels': [label], 'sentences': [sentence]}
            if embeddings:
                words_data[word]['embeddings'] = [embedding]
            if indices:
                words_data[word]['indices'] = [index]

    if only_word and only_word in words_data.keys():
        return words_data[only_word]
    else:
        return words_data


def convert_to_np_array(string_list):
    return np.array([x.split(' ') for x in string_list], dtype=float)


def write_data(out_file, data, mode="w", centroids=None):
    visited_labels = []
    with open(out_file, mode, encoding="utf8") as outf:
        for label, word, sentence in data: #, embedding

            if centroids is not None and label not in visited_labels:
                centroid_string = " ".join([str(x) for x in centroids[label]])
                outf.write("\t".join([str(label), label, centroid_string]) + "\n")
                visited_labels.append(label)

            #embedding_str = " ".join([str(x) for x in embedding])
            outf.write("\t".join([str(label), word, sentence]) + "\n") #, embedding_str


def write_grouped_data(outf, data, centroid=None):
    for label, word, sentence in data: #, embedding

        if centroid is not None:
            centroid_string = " ".join([str(x) for x in centroid])
            outf.write("\t".join([str(label), label, centroid_string]) + "\n")

        #embedding_str = " ".join([str(x) for x in embedding])
        outf.write("\t".join([str(label), word, sentence]) + "\n") #, embedding_str


def write_data_for_classification(outf, data):
    for label, word, centroid in data:
        centroid_string = " ".join([str(x) for x in centroid])
        outf.write("\t".join([str(label), word, centroid_string]) + "\n")


def concatenate_files(file_list, out_file):
    with open(out_file, "w", encoding="utf8") as out:
        for f in file_list:
            with open(f, "r", encoding="utf8") as in_file:
                out.writelines(in_file.readlines())


def remove_duplicate_lines(in_file, out_file, range=None):
    visited_lines = []

    with open(in_file, 'r', encoding="utf8") as input:
        with open(out_file, 'wt', encoding="utf8") as output:

            for line in input.readlines():
                if line not in visited_lines:

                    output.write(line)
                    visited_lines.append(line)
                    if range:
                        visited_lines = visited_lines[-range:]


def sort_lines(in_file, out_file):
    with open(in_file, 'r', encoding="utf8") as input:
        with open(out_file, 'wt', encoding="utf8") as output:

            lines = input.readlines()
            lines.sort(key=lambda x: x.split("\t")[0])

            for line in lines:
                output.write(line)


def count_words(in_file, out_file):
    words = {}
    with open(in_file, 'r', encoding="utf8") as input:
        with open(out_file, 'w', encoding="utf8") as output:
            lines = input.readlines()

            for line in lines:
                word = line.split("\t")[0]

                if word in words.keys():
                    words[word] += 1
                else:
                    words[word] = 1

            output.write("\n".join(["%s %d" % (k, v) for k, v in words.items()]))


def create_sample_word_file(file, word, out_file):
    with open(file, "r", encoding="utf8") as f:
        with open(out_file, "w", encoding="utf8") as outf:

            for line in f:
                data_line = line.strip().split("\t")
                data_word = data_line[0]

                if data_word != word:
                    continue
                else:
                    outf.write(line)


def filter_file_by_words(file, words_file, out_file, word_idx=0, split_by="\t", complement=False, skip_idx=None):
    words = get_unique_words(words_file)

    with open(file, "r", encoding="utf8") as f:
        with open(out_file, "w", encoding="utf8") as outf:

            for line in f:
                word = line.split(split_by)[word_idx]

                if (complement and word not in words) or (not complement and word in words):
                    if skip_idx:
                        split_line = line.split(split_by)
                        outf.write(split_by.join([x for i, x in enumerate(split_line) if i != skip_idx]))
                    else:
                        outf.write(line)


def get_random_part(in_file, out_file1, out_file2, out_file_words1, out_file_words, ratio=0.5):
    n = file_len(in_file)
    n_part = n * ratio

    words_data = load_validation_file_grouped(in_file, all_strings=True)
    words_count = [(key, len(words_data[key]['sentences'])) for key in words_data.keys()]
    shuffle(words_count)

    sum = 0
    idx = 0
    words_part = []

    while sum < n_part:
        word, count = words_count[idx]
        words_part.append(word)
        sum += count
        idx += 1

    words_part.sort()
    with open(out_file_words, "w", encoding="utf8") as out_words:
        out_words.writelines(["\n".join(words_part)])

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



if __name__ == '__main__':
    pass
