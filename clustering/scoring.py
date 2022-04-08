from itertools import permutations
from statistics import mean, stdev

from scipy.stats import moment, gmean
from sklearn import metrics


def get_avg_scores(score_per_word, score_names):
    scores = {}

    for name in score_names:
        non_null_scores = [word_score[name] for word_score in score_per_word.values() if name in word_score.keys() and word_score[name] != None]
        scores[name] = get_avg_score(non_null_scores)

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

    return adj_rand_score, completeness_score


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
    #elif len(list1) > len(list2):
    #    #mappings = [list(zip(list1, p)) for p in combinations_with_replacement(list2, len(list1))]
    #    #return [f for f in mappings if set([x[1] for x in f]) == set(list2)]
    else:
        return None
        #mappings = [list(zip(list1, p)) for p in permutations(list2, len(list1))]
        #mappings = [f for f in mappings if set([x[0] for x in f]) == set(list1)]
        #return mappings

def unsupervised_cluster_score(embeddings, labels):
    silhouette = metrics.silhouette_score(embeddings, labels, metric='cosine')
    db_score = metrics.davies_bouldin_score(embeddings, labels)
    ch_score = metrics.calinski_harabasz_score(embeddings, labels)

    data_moments = get_moments(embeddings)

    clusters = {label: [] for label in labels}
    for label, embedding in zip(labels, embeddings):
        clusters[label].append(embedding)

    moments = {label: get_moments(clusters[label]) for label in clusters.keys()}
    avg_moments = {}

    moment_names = ["mean", "variance", "stddev", "skewness", "kurtosis", "6th", "7th", "8th"]
    for m in moment_names:
        moment_all_labels = [moments[l][m] for l in labels]
        avg_moments[m] = gmean(moment_all_labels)

    return silhouette, db_score, ch_score, data_moments, avg_moments

def get_moments(data):
    moment_names = ["mean", "variance", "stddev", "skewness", "kurtosis", "6th", "7th", "8th"]
    return {m: gmean(moment(data, moment=i+1)) for i, m in enumerate(moment_names)}
