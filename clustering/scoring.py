import os
from itertools import permutations, combinations_with_replacement
from statistics import stdev, mean
from sklearn import metrics

import file_helpers


def get_avg_scores(score_per_word, score_names):
    scores = {}

    for name in score_names:
        non_null_scores = [word_score[name] for word_score in score_per_word.values() if name in word_score.keys() and word_score[name] != None]
        """non_null_scores = []
        for n_clusters_score in score_per_word.values():
            for word_score in n_clusters_score.values():
                print(name, list(word_score.keys()))
                if name in word_score.keys():# and word_score[name] != None:
                    non_null_scores.append(word_score[name])"""
        scores[name] = get_avg_score(non_null_scores)
        """for n_clusters in score_per_word.keys():
            avg_score = get_avg_score([x[name] for x in score_per_word[n_clusters].values() if x[name]])

            if name in scores.keys():
                scores[name][n_clusters] = avg_score
            else:
                scores[name] = {n_clusters: avg_score}"""

    return scores

def get_avg_score(score_per_word):
    if len(score_per_word) == 0:
        return None, None, 0
    elif len(score_per_word) == 1:
        return score_per_word[0], None, 1
    return mean(score_per_word), stdev(score_per_word), len(score_per_word)


def score_clustering(predicted_labels, validation_labels):

    adj_rand_score = metrics.adjusted_rand_score(validation_labels, predicted_labels)
    completeness_score = metrics.completeness_score(validation_labels, predicted_labels)
    f1_score, labels, confusion_matrix = f1_confusion_matrix(predicted_labels, validation_labels)

    return adj_rand_score, completeness_score, f1_score, labels, confusion_matrix


def f1_confusion_matrix(predicted_labels, validation_labels):
    validation_labels = [int(i) for i in validation_labels]
    score, labels = 0, validation_labels
    labels1, labels2 = list(set(predicted_labels)), list(set(validation_labels))

    if len(labels1) > 8:
        return None, None, None
    mappings = all_mappings(labels1, labels2)
    if mappings is None or len(mappings) > 1000:
        return None, None, None

    for mapping in mappings:
        dictionary = dict(mapping)
        pred_labels = [dictionary[x] for x in predicted_labels]

        new_score = metrics.f1_score(pred_labels, validation_labels, average='macro')

        if new_score > score:
            score = new_score
            labels = pred_labels

    matrix = metrics.confusion_matrix(predicted_labels, labels)
    return score, labels, matrix

def all_mappings(list1, list2):
    if len(list1) == len(list2):
        return [zip(list1, p) for p in permutations(list2)]
    elif len(list1) > len(list2):
        mappings = [list(zip(list1, p)) for p in combinations_with_replacement(list2, len(list1))]
        return [f for f in mappings if set([x[1] for x in f]) == set(list2)]
    else:
        mappings = [list(zip(list1, p)) for p in permutations(list2, len(list1))]
        mappings = [f for f in mappings if set([x[0] for x in f]) == set(list1)]
        return mappings


def compare_clusters(words, cluster_files, clustering_names, validation_file, out_file):
    if isinstance(words, str):
        words = file_helpers.get_unique_words(words)
    else:
        words = words

    print("loading data ...")
    val_data = file_helpers.load_validation_file_grouped(validation_file)

    words = [w for w in words if w in val_data.keys() and len(set(val_data[w]['labels'])) > 1]

    cluster_data = []
    for file in cluster_files:
        print("Filtering file %s ..." % file)
        new_file = os.path.join(os.path.dirname(file), "clusters.tsv")
        file_helpers.filter_file_by_words(file, words, new_file, word_idx=1, skip_idx=3)

        data = file_helpers.load_file(new_file)
        new_data = {w:{} for w in words}

        for label, word, sentence in data:
            new_data[word][sentence] = label

        cluster_data.append(new_data)

    print("writing data ...")

    with open(out_file, "w", encoding="utf8") as f:
        for word in words:
            f.write((37 + len(word)) * "*" + "\n")
            f.write("*** Cluster comparison for word: %s ***\n" % word)
            f.write((37 + len(word)) * "*" + "\n")
            f.write("%s\tValidation data\tSentence\n" % "\t".join(clustering_names))

            for i, sentence in enumerate(val_data[word]['sentences']):
                predicted = [x[word][sentence] if sentence in x[word] else "None" for x in cluster_data]
                #if not(len(set(predicted)) == 1 and predicted.pop() == "None"):
                f.write("\t".join(predicted + [val_data[word]['labels'][i], sentence]) + "\n")
