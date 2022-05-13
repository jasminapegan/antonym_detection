from scipy.stats import mode, entropy
from scipy.spatial.distance import cosine
from io import TextIOWrapper
from math import floor
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import file_helpers

class SenseClusters():

    def __init__(self, syn_ant_file: str,  cluster_file: str, sense_file: str,  out_file: str, out_sentences: str=None,
                 k: int=1, algo: str='avg_dist', clean_data: bool=True, ratio=0.05, weights='distance'):
        self.cluster_data = file_helpers.load_validation_file_grouped(cluster_file, embeddings=True)
        self.sense_data = file_helpers.words_data_to_dict(sense_file, header=False, skip_num=False)
        self.classifier = KNeighborsClassifier(n_neighbors=k, metric='cosine', weights=weights)
        self.n = file_helpers.file_len(syn_ant_file)

        assert algo in ['avg_dist', 'weighed_entropy']
        self.algo = algo

        if out_sentences:
            self.out_sentences = open(out_sentences, "w", encoding="utf8")
        else:
            self.out_sentences = None

        with open(out_file, "w", encoding='utf8') as outf:
            with open(syn_ant_file, "r", encoding='utf8') as f:
                for i, line in enumerate(f):

                    if i % 2000 == 0:
                        print(i, "/", self.n)

                    w1, w2, label = line.split("\t")
                    label = int(label)

                    if label != 0 and label != 1:
                        continue

                    if w1 not in self.cluster_data.keys() or w2 not in self.cluster_data.keys():
                        continue

                    self.word_data1 = self.cluster_data[w1]
                    self.word_data2 = self.cluster_data[w2]

                    if clean_data:
                        self.word_data1 = remove_outliers(self.word_data1, ratio=ratio)
                        self.word_data2 = remove_outliers(self.word_data2, ratio=ratio)

                    self.sense_data1 = self.sense_data[w1]
                    self.sense_data2 = self.sense_data[w2]

                    n1, n2 = len(set(self.word_data1['labels'])), len(set(self.word_data2['labels']))
                    if n1 == len(self.sense_data1) and n2 == len(self.sense_data2):
                        ant_syn = "antonym" if label == 0 else "synonym"
                        out_line = f"Data for {ant_syn} pair '{w1}' - '{w2}'\n"

                        if self.out_sentences:
                            self.out_sentences.write(out_line)

                        outf.write(out_line)
                        self.find_senses(w1, w2, outf, algo)

        if self.out_sentences:
            self.out_sentences.close()

    def find_senses(self, w1: str, w2: str, outf: TextIOWrapper, algo: str):
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

        if algo == 'weighed_entropy':
            self.get_sense_by_entropy(outf)
        elif algo == 'avg_dist':
            self.get_sense_by_avg_dist(w1, w2, outf)

    def get_sense_by_avg_dist(self, w1: str, w2: str, outf):
        sentences1 = get_sentences_by_label(self.word_data1)
        sentences2 = get_sentences_by_label(self.word_data2)

        labels1 = list(set(self.word_data1['labels']))
        labels2 = list(set(self.word_data2['labels']))

        X1_by_label = get_elements_by_label_dict(self.X1, self.word_data1['labels'])
        X2_by_label = get_elements_by_label_dict(self.X2, self.word_data2['labels'])

        avg_dists = calculate_avg_distance(labels1, labels2, X1_by_label, X2_by_label)
        """
        for sense_id1 in labels1:
            avg_dist_row = {label:  for label in labels2}

            for sense_id2 in labels2:

                avg_dist_row[sense_id2] = np.mean([
                    np.mean([cosine(x1, x2) for x2 in X2_by_label[sense_id2]
                             ]) for x1 in X1_by_label[sense_id1]])

            avg_dists[sense_id1] = avg_dist_row"""

        best_scores = {x: [] for x in labels2}

        for label1 in labels1:
            sorted_dist_list = sorted(list(avg_dists[label1].items()), key=lambda x: x[1])
            label2 = sorted_dist_list[0][0]
            score = avg_dists[label1][label2]

            d1 = get_description(self.sense_data1, label1)
            d2 = get_description(self.sense_data2, label2)
            best_scores[label2].append((score, d1, d2, label1))

        min_dist, min_label1, min_label2 = 10 ** 10, None, None
        for label2 in labels2:
            if best_scores[label2]:
                score, d1, d2, label1 = sorted(best_scores[label2], key=lambda x: x[0])[0]

                if score < min_dist:
                    min_dist, min_label1, min_label2 = score, label1, label2

                outf.write(f"Score: {score}\n\tDescriptions: {d1} | {d2}\n")
                outf.write(f"\tExample: {sentences1[label1][0]} | {sentences2[label2][0]}\n")

                if self.out_sentences:
                    self.write_sentences(w1, w2, sentences1[label1], sentences2[label2])

        d1 = get_description(self.sense_data1, min_label1)
        d2 = get_description(self.sense_data2, min_label2)
        outf.write(f"Min score: {min_dist}\n\tDescriptions: {d1} | {d2}\n")

        if min_label1 != None and min_label2 != None:
            outf.write(f"\tExample: {sentences1[min_label1][0]} | {sentences2[min_label2][0]}\n\n")

    def write_sentences(self, w1, w2, s1, s2):
        self.out_sentences.write(f"{w1}\n")
        for s in s1:
            self.out_sentences.write(f"\t{s}\n")
        self.out_sentences.write(f"{w2}\n")
        for s in s2:
            self.out_sentences.write(f"\t{s}\n")
        self.out_sentences.write("\n")

    def get_sense_by_entropy(self, outf):
        sentences1 = get_sentences_by_label(self.word_data1)
        sentences2 = get_sentences_by_label(self.word_data2)

        min_e, label1, label2 = 10**10, None, None

        for sense_id in set(self.word_data2['labels']):
            y = get_elements_by_label(self.y2, self.word_data2['labels'], sense_id)

            if len(y) > 0:
                mode_idx = str(mode(y)[0][0])

                e = (1 + entropy(y)) / np.log(len(y))
                if e < min_e:
                    min_e, label1, label2 = e, mode_idx, sense_id

        outf.write(f"Min entropy: {min_e} | {get_description(self.sense_data1, label1)} | {get_description(self.sense_data2, label2)}\n")
        if label1 != None and label2 != None:
            outf.write(f"Example: {sentences1[label1][0]} | {sentences2[label2][0]}\n\n")


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

        data_mean, data_std = np.mean(X), np.std(X)
        cut_off = data_std * 3
        lower, upper = data_mean - cut_off, data_mean + cut_off

        i += 1

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