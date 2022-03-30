from itertools import product
from statistics import stdev, mean
from sklearn import metrics


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
    #f1_score, labels, confusion_matrix = f1_confusion_matrix(predicted_labels, validation_labels)

    return adj_rand_score, completeness_score #f1_score, labels, confusion_matrix


def f1_confusion_matrix(predicted_labels, validation_labels):
    validation_labels = [int(i) for i in validation_labels]
    score, labels = 0, validation_labels
    labels1, labels2 = list(set(predicted_labels)), list(set(validation_labels))

    if len(labels1) != len(labels2):
        return None, None, None

    for i in product(*([labels2] * len(labels1))):
        mapping = dict(zip(labels1, i))
        pred_labels = [mapping[x] for x in predicted_labels]

        new_score = metrics.f1_score(pred_labels, validation_labels, average='macro')

        if new_score > score:
            score = new_score
            labels = pred_labels

    matrix = metrics.confusion_matrix(predicted_labels, labels)
    return score, labels, matrix

