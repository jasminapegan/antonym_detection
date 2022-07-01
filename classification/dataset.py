import itertools
from random import shuffle

import classla
import numpy as np
from clustering.scoring import parse_cluster_file, parse_score_data

import file_helpers
from data.embeddings import WordEmbeddings


def create_dataset(cluster_file, score_file_ant, score_file_syn, examples_file, out_file_syn, out_file_ant,
                   out_anti_syn, out_anti_ant, out_syn_test, out_ant_test, out_syn_train, out_ant_train):
    clusters = parse_cluster_file(cluster_file)
    score_data_ant = parse_score_data(score_file_ant)
    score_data_syn = parse_score_data(score_file_syn)
    score_data = {x: ('ant', y) for x, y in score_data_ant.items()}
    score_data.update({x: ('syn', y) for x, y in score_data_syn.items()})
    examples_data = file_helpers.load_validation_file_grouped(examples_file, indices=True, embeddings=False, use_pos=False)

    f = open(out_file_syn, "w", encoding="utf8")
    g = open(out_file_ant, "w", encoding="utf8")
    f2 = open(out_anti_syn, "w", encoding="utf8")
    g2 = open(out_anti_ant, "w", encoding="utf8")

    for pair_data in clusters.keys():

        cluster = clusters[pair_data]
        ant_syn, is_correct = score_data[pair_data]

        if "DA" in is_correct:
            #w1, w2, pos = pair_data
            w1, w2 = pair_data
            sense_w1, sense_w2 = cluster["w1_sense"], cluster["w2_sense"]
            pos1, pos2 = list(examples_data[w1].keys())[0], list(examples_data[w2].keys())[0]
            if pos1 != pos2:
                continue
            #w1_data, w2_data = examples_data[w1]["all"], examples_data[w2]["all"] #[pos]
            w1_data, w2_data = examples_data[w1][pos1], examples_data[w2][pos2] #[pos]

            if ant_syn == "ant":
                write_data_to_file(g, w1, w2, sense_w1, sense_w2, w1_data, w2_data)
                write_other_pairs(g2, w1, w2, sense_w1, sense_w2, w1_data, w2_data)
            else:
                write_data_to_file(f, w1, w2, sense_w1, sense_w2, w1_data, w2_data)
                write_other_pairs(f2, w1, w2, sense_w1, sense_w2, w1_data, w2_data)

    join_to_dataset(out_file_syn, out_anti_syn, out_syn_test, out_syn_train)
    join_to_dataset(out_file_ant, out_anti_ant, out_ant_test, out_ant_train)

def write_data_to_file(file, w1, w2, sense_w1, sense_w2, w1_data, w2_data):
    w1_examples = [(s, i) for l, s, i in zip(w1_data['labels'], w1_data['sentences'], w1_data['indices']) if
                   l == sense_w1]
    w2_examples = [(s, i) for l, s, i in zip(w2_data['labels'], w2_data['sentences'], w2_data['indices']) if
                   l == sense_w2]
    # w1, pos1, form1, l1, idx1, s1, w2, pos2, form2, l2, idx2, s2, (label)
    for s_i_1, s_i_2 in itertools.product(w1_examples, w2_examples):
        s1, i1 = s_i_1
        s2, i2 = s_i_2
        f1, f2 = s1.split(" ")[int(i1)], s2.split(" ")[int(i2)]

        file.write(f"{w1}\t/\t{f1}\t{sense_w1}\t{i1}\t{s1}\t")
        file.write(f"{w2}\t/\t{f2}\t{sense_w2}\t{i2}\t{s2}\t1\n")

def write_other_pairs(file, w1, w2, sense_w1, sense_w2, w1_data, w2_data):
    w1_examples = [(s, i) for l, s, i in zip(w1_data['labels'], w1_data['sentences'], w1_data['indices']) if
                   l == sense_w1]
    w2_examples = [(s, i) for l, s, i in zip(w2_data['labels'], w2_data['sentences'], w2_data['indices']) if
                   l == sense_w2]
    w1_anti_examples = [(s, i) for l, s, i in zip(w1_data['labels'], w1_data['sentences'], w1_data['indices']) if
                   l != sense_w1]
    w2_anti_examples = [(s, i) for l, s, i in zip(w2_data['labels'], w2_data['sentences'], w2_data['indices']) if
                   l != sense_w2]
    # w1, pos1, form1, l1, idx1, s1, w2, pos2, form2, l2, idx2, s2, (label)
    for s_i_1, s_i_2 in itertools.product(w1_examples, w2_anti_examples):
        s1, i1 = s_i_1
        s2, i2 = s_i_2
        f1, f2 = s1.split(" ")[int(i1)], s2.split(" ")[int(i2)]

        file.write(f"{w1}\t/\t{f1}\t{sense_w1}\t{i1}\t{s1}\t")
        file.write(f"{w2}\t/\t{f2}\t{sense_w2}\t{i2}\t{s2}\t1\n")

    for s_i_1, s_i_2 in itertools.product(w1_anti_examples, w2_examples):
        s1, i1 = s_i_1
        s2, i2 = s_i_2
        f1, f2 = s1.split(" ")[int(i1)], s2.split(" ")[int(i2)]

        file.write(f"{w1}\t/\t{f1}\t{sense_w1}\t{i1}\t{s1}\t")
        file.write(f"{w2}\t/\t{f2}\t{sense_w2}\t{i2}\t{s2}\t1\n")

def join_to_dataset(f1, f2, outf_test, outf_train, split=0.2):
    n1, n2 = file_helpers.file_len(f1), file_helpers.file_len(f2)
    m = min(n1, n2)

    with open(f1, "r", encoding="utf8") as f:
        lines1 = f.readlines()

    with open(f2, "r", encoding="utf8") as f:
        lines2 = f.readlines()

    shuffle(lines1)
    shuffle(lines2)

    all_lines = lines1[:m] + lines2[:m]
    shuffle(all_lines)

    r = int(split * m)

    with open(outf_test, "w", encoding="utf8") as f:
        f.writelines(all_lines[:r])

    with open(outf_train, "w", encoding="utf8") as f:
        f.writelines(all_lines[r:])

def get_bert_embeddings(sentence_file, embeddings_out_file, limit=None):
    # w1, pos1, form1, label1, idx1, sentence1, w2, pos2, form2, label2, idx2, sentence2
    we = WordEmbeddings()
    lemmatizer =classla.Pipeline('sl', processors='tokenize,pos,lemma', use_gpu=True)

    with open(sentence_file, "r", encoding="utf8") as f:
        data = f.readlines()

    with open(embeddings_out_file, "w", encoding="utf8") as outf:
        for i, line in enumerate(data):
            w1, pos1, form1, l1, idx1, s1, w2, pos2, form2, l2, idx2, s2, label = parse_line(line, embeddings=False)
            res1, res2 = we.get_words_embeddings_2([w1, w2], [pos1, pos2], [idx1, idx2], [s1, s2], lemmatizer=lemmatizer, lemmatized=False)
            e1 = " ".join([str(float(x)) for x in res1[-1]])
            e2 = " ".join([str(float(x)) for x in res2[-1]])
            outf.write(f"{w1}\t{pos1}\t{form1}\t{idx1}\t{l1}\t{s1}\t{e1}\t")
            outf.write(f"{w2}\t{pos2}\t{form2}\t{idx2}\t{l2}\t{s2}\t{e2}\t")
            outf.write(f"{label}\n")

            if limit and i >= limit:
                break

def parse_line(line, embeddings=False, only_embeddings=False, labels=True):
    # w1, pos1, form1, l1, idx1, s1, e1, w2, pos2, form2, l2, idx2, s2, e2, (label)
    data = line.strip().split("\t")

    if not embeddings:
        # w1, pos1, form1, l1, idx1, s1, w2, pos2, form2, l2, idx2, s2, (label)
        assert 12 <= len(data) <= 13
        data[4], data[10] = int(data[4]), int(data[10])
        return data
    else:
        # w1, pos1, form1, l1, idx1, s1, e1, w2, pos2, form2, l2, idx2, s2, e2, (label)
        data[4], data[11] = int(data[4]), int(data[11])
        e1 = [float(x) for x in data[6].split(" ")]
        e2 = [float(x) for x in data[13].split(" ")]

        if labels:
            if len(data) != 15:
                return []
            if only_embeddings:
                return e1, e2, data[-1]
            else:
                return data
        else:
            assert len(data) == 14

            if only_embeddings:
                return e1, e2
            else:
                return data

def compare_with_new_data(old_examples, new_examples, words, out_file):
    with open(old_examples, "r", encoding="utf8") as f:
        with open(new_examples, "r", encoding="utf8") as g:
            lines = set(f.readlines())
            new_lines = []

            for line2 in g:
                w2, pos2, wf2, l2, i2, s2 = line2.split("\t")
                line2 = "\t".join([w2, wf2, l2, i2, s2])

                if line2 not in lines:
                    new_lines.append(line2)

        print("New examples:", len(new_lines))
        lines_ant = []

        with open(out_file, "w", encoding="utf8") as f:
            for line in new_lines:
                line_split = line.split("\t")
                word, sense = line_split[0], line_split[2]
                if word in words.keys() and sense == words[word]:
                    lines_ant.append(line)
                    f.write(line)

        print("Filtered new examples:", len(lines_ant))

def parse_embeddings_data(filename, limit_range):
    data, labels = [], []
    with open(filename, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            if not limit_range or i in limit_range:

                data_line = parse_line(line, embeddings=True, only_embeddings=True)
                if not data_line:
                    return
                data.append(data_line[0] + data_line[1])
                labels.append(data_line[-1])

            if limit_range and i > limit_range[-1]:
                break

    data = np.array(data).astype('float32')
    print("data shape: ", data.shape)

    labels = np.array(labels).astype('int')
    return data, labels

def parse_sentence_data(filename, limit_range):
    data = {'labels': [], 'sentence_pairs': [], 'index_pairs': []}
    with open(filename, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            if not limit_range or i in range(limit_range):

                w1, pos1, form1, l1, idx1, s1, w2, pos2, form2, l2, idx2, s2, label = parse_line(line, embeddings=True, only_embeddings=True)

                data['word_pairs'].append((w1, w2))
                data['form_pairs'].append((form1, form2))
                data['index_pairs'].append((idx1, idx2))
                data['sentence_pairs'].append((s1, s2))
                data['labels'].append(label)

            if limit_range and i > limit_range[1]:
                break

    data['labels'] = np.array(data['labels']).astype('int')
    return data