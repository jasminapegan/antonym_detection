import re

from scipy.spatial.distance import cosine
from io import TextIOWrapper
from math import floor
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import file_helpers

class SenseClusters():

    def __init__(self, syn_ant_file: str,  cluster_file: str, sense_file: str,  out_file: str, out_sentences: str=None,
                 k: int=1, algo: str='avg_dist', clean_data: bool=True, ratio: float=0.05, weights='distance'):
        print(f"Reading data ...")

        self.cluster_data = file_helpers.load_validation_file_grouped(cluster_file, embeddings=True)
        self.sense_data = file_helpers.words_data_to_dict(sense_file, header=False, skip_num=False)
        self.classifier = KNeighborsClassifier(n_neighbors=k, metric='cosine', weights=weights)

        self.set_algorithm(algo)

        if out_sentences:
            self.out_sentences = open(out_sentences, "w", encoding="utf8")
        else:
            self.out_sentences = None

        with open(syn_ant_file, "r", encoding='utf8') as f:
            self.lines = f.readlines()

        self.execute_algorithm(out_file, algo=algo, clean_data=clean_data, ratio=ratio)

        if self.out_sentences:
            self.out_sentences.close()

    def set_algorithm(self, algo):
        assert algo in ['avg_dist', 'min_dist', 'avg_min_dist', 'min_avg_dist', 'max_dist', 'max_avg_dist', 'avg_max_dist']
        self.algo = algo

    def execute_algorithm(self, out_file: str, algo: str='avg_dist', clean_data: bool=False, ratio: float=0.05):
        print(f"Initiating algorithm {algo} ...")

        self.set_algorithm(algo)
        self.ratio = ratio
        self.results = []

        with open(out_file, "w", encoding='utf8') as outf:
            for i, line in enumerate(self.lines):

                if i % 1000 == 0:
                    print(i)

                self.w1, self.w2, label = line.split("\t")

                if self.w1 == self.w2:
                    continue

                self.label = int(label)

                if self.label != 0 and self.label != 1:
                    continue

                if self.w1 not in self.cluster_data.keys() or self.w2 not in self.cluster_data.keys():
                    continue

                pos_tags1 = self.cluster_data[self.w1].keys()
                pos_tags2 = self.cluster_data[self.w2].keys()

                for pos in set(pos_tags1).intersection(pos_tags2):
                    self.pos = pos
                    self.find_clusters_for_pair(clean_data, outf)

    def find_clusters_for_pair(self, clean_data, outf):
        self.word_data1, self.word_data2 = self.cluster_data[self.w1][self.pos], self.cluster_data[self.w2][self.pos]
        word_data_labels_1 = sorted(list(set([int(x) for x in self.word_data1['labels']])))
        word_data_labels_2 = sorted(list(set([int(x) for x in self.word_data2['labels']])))

        # check there is no weird gap between sense labels
        if not self.check_senses(word_data_labels_1, word_data_labels_2):
            return

        if clean_data:
            self.word_data1 = remove_outliers(self.word_data1, ratio=self.ratio)
            self.word_data2 = remove_outliers(self.word_data2, ratio=self.ratio)

        self.sense_data1, self.sense_data2 = self.sense_data[self.w1], self.sense_data[self.w2]
        sense_data_labels_1 = sorted(list(set([int(x['num']) for x in self.sense_data1])))
        sense_data_labels_2 = sorted(list(set([int(x['num']) for x in self.sense_data2])))

        # every sense needs at least one example
        if not self.check_examples(word_data_labels_1, word_data_labels_2, sense_data_labels_1, sense_data_labels_2):
            return

        ant_syn = "antonym" if self.label == 0 else "synonym"

        if self.out_sentences:
            self.out_sentences.write(f"Data for {ant_syn} pair {self.w1} - {self.w2} (POS {self.pos})\n")

        out_lines = [f"Data for {ant_syn} pair {self.w1} - {self.w2} (POS {self.pos})\n"]
        out_lines += [f"Sense data for {self.w1}\nSense\tdescription\n"]
        out_lines += [f"\t{s['num']}\t{s['description']}\n" for s in self.sense_data1]
        out_lines += [f"Sense data for {self.w2}:\nSense\t description\n"]
        out_lines += [f"\t{s['num']}\t{s['description']}\n" for s in self.sense_data2]

        outf.writelines(out_lines)

        self.fit_predict(outf)
        self.find_senses(outf)

    def fit_predict(self, outf: TextIOWrapper):
        self.X1 = file_helpers.convert_to_np_array(self.word_data1['embeddings'])
        self.X1 = fix_array_shape(self.X1)
        self.y1 = np.array([int(x) for x in self.word_data1['labels']])

        self.classifier.fit(self.X1, self.y1)

        self.X2 = file_helpers.convert_to_np_array(self.word_data2['embeddings'])
        self.X2 = fix_array_shape(self.X2)

        try:
            self.y2 = self.classifier.predict(self.X2)
        except Exception as e:
            outf.write("Error: %s\n" % e)
            return

    def find_senses(self, outf: TextIOWrapper):
        sentences1 = get_sentences_by_label(self.word_data1)
        sentences2 = get_sentences_by_label(self.word_data2)

        labels1 = list(set(self.word_data1['labels']))
        labels2 = list(set(self.word_data2['labels']))

        X1_by_label = get_elements_by_label_dict(self.X1, self.word_data1['labels'])
        X2_by_label = get_elements_by_label_dict(self.X2, self.word_data2['labels'])

        if self.algo == 'avg_dist':
            dists = calculate_avg_distance(labels1, labels2, X1_by_label, X2_by_label)
        elif self.algo == 'min_dist':
            dists = calculate_min_distance(labels1, labels2, X1_by_label, X2_by_label)
        elif self.algo == 'avg_min_dist':
            dists = calculate_avg_min_distance(labels1, labels2, X1_by_label, X2_by_label)
        elif self.algo == 'min_avg_dist':
            dists = calculate_min_avg_distance(labels1, labels2, X1_by_label, X2_by_label)
        elif self.algo == 'max_dist':
            dists = calculate_max_distance(labels1, labels2, X1_by_label, X2_by_label)
        elif self.algo == 'max_avg_dist':
            dists = calculate_max_avg_distance(labels1, labels2, X1_by_label, X2_by_label)
        elif self.algo == 'avg_max_dist':
            dists = calculate_avg_max_distance(labels1, labels2, X1_by_label, X2_by_label)

        self.X1_by_label = X1_by_label
        self.X2_by_label = X2_by_label

        self.write_output(labels1, labels2, sentences1, sentences2, dists, outf)

    def write_sentences(self, s1, s2):
        self.out_sentences.write(f"{self.w1}\n")
        for s in s1:
            self.out_sentences.write(f"\t{s}\n")
        self.out_sentences.write(f"{self.w2}\n")
        for s in s2:
            self.out_sentences.write(f"\t{s}\n")
        self.out_sentences.write("\n")

    def write_output(self, labels1, labels2, sentences1, sentences2, dists, outf):
        best_scores = {x: [] for x in labels2}

        for label1 in labels1:

            sorted_dist_list = sorted(list(dists[label1].items()), key=lambda x: x[1])
            label2 = sorted_dist_list[0][0]
            score = dists[label1][label2]

            d1 = get_description(self.sense_data1, label1)
            d2 = get_description(self.sense_data2, label2)
            s1 = get_elements_by_label(self.word_data1, labels1, label1)
            s2 = get_elements_by_label(self.word_data1, labels2, label2)
            best_scores[label2].append((score, d1, d2, s1, s2, label1))

        min_dist, min_label1, min_label2, min_s1, min_s2 = 10 ** 10, None, None, None, None
        for label2 in labels2:
            if best_scores[label2]:
                score, d1, d2, s1, s2, label1 = sorted(best_scores[label2], key=lambda x: x[0])[0]

                if score < min_dist:
                    min_dist, min_label1, min_label2, min_s1, min_s2 = score, label1, label2, s1, s2

                #outf.write(f"Score: {score}\n")
                #outf.write(f"\tExample: {sentences1[label1][0]} | {sentences2[label2][0]}\n")

                if self.out_sentences:
                    self.write_sentences(sentences1[label1], sentences2[label2])

        outf.write(f"Predicted sense pair: {self.w1}({min_label1}) and {self.w2}({min_label2}) " +
                   f"with distance score: {min_dist}\n")

        if min_label1 != None and min_label2 != None:
            outf.write(f"{self.w1}({min_label1}): {sentences1[min_label1][0]}\n")
            outf.write(f"{self.w2}({min_label2}): {sentences2[min_label2][0]}\n\n")

            self.results.append({"w1": self.w1, "w2": self.w2,
                                 "sentences1": sentences1[min_label1], "sentences2": sentences2[min_label2],
                                 "embeddings1": self.X1_by_label[min_label1], "embeddings2": self.X2_by_label[min_label2]})

    def check_senses(self, word_data_labels_1, word_data_labels_2):
        diff1 = set(word_data_labels_1).difference(set(range(len(word_data_labels_1))))
        diff2 = set(word_data_labels_2).difference(set(range(len(word_data_labels_2))))
        if len(diff1) > 0:
            descriptions = {int(x['num']): x['description'] for x in self.sense_data[self.w1]}
            missing = [descriptions[d].lower() for d in diff1]
            if set(missing) == {"nov pomen"}:
                pass
            else:
                print(f"word {self.w1} (POS {self.pos}) - missing some sense examples for senses: {descriptions}")
                return False
        if len(diff2) > 0:
            descriptions = {int(x['num']): x['description'] for x in self.sense_data[self.w2]}
            missing = [descriptions[d].lower() for d in diff2]
            if set(missing) == {"nov pomen"}:
                pass
            else:
                print(f"word {self.w2} (POS {self.pos}) - missing some sense examples for senses: {descriptions}")
                return False
        return True

    def check_examples(self, word_data_labels_1, word_data_labels_2, sense_data_labels_1, sense_data_labels_2):
        diff1 = set(sense_data_labels_1).difference(set(word_data_labels_1))
        diff2 = set(sense_data_labels_2).difference(set(word_data_labels_2))

        if len(diff1) > 0:
                print(f"word {self.w1} (POS {self.pos}) missing {len(diff1)} examples: {diff1}")
                return False
        if len(diff2) > 0:
                print(f"word {self.w2} (POS {self.pos}) missing {len(diff2)} examples: {diff2}")
                return False

        return True

def get_sentences_by_label(word_data):
    sentences = {i: [] for i in set(word_data['labels'])}
    for x, i in zip(word_data['sentences'], word_data['labels']):
        sentences[i].append(x)
    return sentences

def fix_array_shape(X):
    if len(X.shape) == 1:
        X = np.array(X)
        X.reshape(-1, 1)
    return X

def get_elements_by_label_dict(data, labels):
    return {label: get_elements_by_label(data, labels, label) for label in labels}

def get_elements_by_label(data, labels, label):
    return [x for x, i in zip(data, labels) if i == label]

def get_description(sense_data, sense_id):
    elements = [x for x in sense_data if x['num'] == sense_id]
    if elements:
        return elements[0]['description']
    else:
        return "/"

def remove_outliers(word_data, ratio=0.05):
    data = list(zip(np.array(word_data['embeddings']), word_data['labels'], word_data['sentences']))
    new_data = []

    labels = list(set(word_data['labels']))
    for label in labels:
        data_filtered = [x for x in data if x[1] == label]
        new_data.append(remove_outliers_from_one_sense(data_filtered, ratio=ratio))

    embeddings = [d[0] for d in data]
    labels = [d[1] for d in data]
    sentences = [d[2] for d in data]

    return {'embeddings': embeddings, 'labels': labels, 'sentences': sentences}

def remove_outliers_from_one_sense(data, ratio):
    n = len(data)
    if n < 5:
        return data

    outlier, lower, upper = 0, 0, 0
    limit = floor(ratio * n)
    i = 0

    while i < limit and np.all(outlier < lower) or np.all(outlier > upper):
        X = sorted(data, key=lambda x: data_mean - x[0])
        outlier = X[-1][0]
        i += 1

        data_mean, data_std = np.mean(X), np.std(X)
        cut_off = data_std * 3
        lower, upper = data_mean - cut_off, data_mean + cut_off

    return data

def calculate_avg_distance(labels1, labels2, X1_by_label, X2_by_label):
    return {label1:
                {label2:
                     np.mean([
                         np.mean([cosine(x1, x2) for x2 in X2_by_label[label2]])
                         for x1 in X1_by_label[label1]])
                 for label2 in labels2}
            for label1 in labels1}

def calculate_min_distance(labels1, labels2, X1_by_label, X2_by_label):
    return {label1:
                {label2:
                     np.min([
                         np.min([cosine(x1, x2) for x2 in X2_by_label[label2]])
                         for x1 in X1_by_label[label1]])
                 for label2 in labels2}
            for label1 in labels1}

def calculate_min_avg_distance(labels1, labels2, X1_by_label, X2_by_label):
    return {label1:
                {label2:
                     np.min([
                         np.mean([cosine(x1, x2) for x2 in X2_by_label[label2]])
                         for x1 in X1_by_label[label1]])
                 for label2 in labels2}
            for label1 in labels1}

def calculate_avg_min_distance(labels1, labels2, X1_by_label, X2_by_label):
    return {label1:
                {label2:
                     np.mean([
                         np.min([cosine(x1, x2) for x2 in X2_by_label[label2]])
                         for x1 in X1_by_label[label1]])
                 for label2 in labels2}
            for label1 in labels1}

def calculate_max_distance(labels1, labels2, X1_by_label, X2_by_label):
    return {label1:
                {label2:
                     np.max([
                         np.max([cosine(x1, x2) for x2 in X2_by_label[label2]])
                         for x1 in X1_by_label[label1]])
                 for label2 in labels2}
            for label1 in labels1}

def calculate_max_avg_distance(labels1, labels2, X1_by_label, X2_by_label):
    return {label1:
                {label2:
                     np.max([
                         np.mean([cosine(x1, x2) for x2 in X2_by_label[label2]])
                         for x1 in X1_by_label[label1]])
                 for label2 in labels2}
            for label1 in labels1}

def calculate_avg_max_distance(labels1, labels2, X1_by_label, X2_by_label):
    return {label1:
                {label2:
                     np.mean([
                         np.max([cosine(x1, x2) for x2 in X2_by_label[label2]])
                         for x1 in X1_by_label[label1]])
                 for label2 in labels2}
            for label1 in labels1}

def print_missing_senses(sense_data_file, sense_examples_file, word_data):
    cluster_data = file_helpers.load_validation_file_grouped(sense_examples_file, embeddings=True)
    sense_data = file_helpers.words_data_to_dict(sense_data_file, header=False, skip_num=False)

    with open(word_data, "r", encoding="utf8") as f:
        text = f.readlines()

    for line in text:
        matches = re.match("pair (.*)-(.*) \(POS (.*)\)", line)

        if not matches:
            continue

        w1, w2, pos = matches.groups()

        print(f"Data for pair {w1}-{w2} (POS {pos})")

        print(f"sense data for {w1}")
        for sense in sense_data[w1]:
            print(f"\t{sense['num']}\t{sense['description']}")

        print(f"sense data for {w2}")
        for sense in sense_data[w2]:
            print(f"\t{sense['num']}\t{sense['description']}")

        print(f"Number of examples for {w1}")
        labels = cluster_data[w1][pos]['labels']
        for label in set(labels):
            print(f"\t{label}\t{labels.count(label)}")

        print(f"Number of examples for {w2}")
        labels = cluster_data[w2][pos]['labels']
        for label in sorted(list(set(labels)), key=lambda x: int(x)):
            print(f"\t{label}\t{labels.count(label)}")

        print()
