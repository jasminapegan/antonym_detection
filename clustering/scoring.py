import re
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
    silhouette = 'nan'
    db_score = 'nan'
    ch_score = 'nan'

    try:
        silhouette = metrics.silhouette_score(embeddings, labels, metric='cosine')
    except Exception as e:
        print(e)
    try:
        db_score = metrics.davies_bouldin_score(embeddings, labels)
    except Exception as e:
        print(e)
    try:
        ch_score = metrics.calinski_harabasz_score(embeddings, labels)
    except Exception as e:
        print(e)

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

def evaluate_cluster_results(in_cluster_old, in_score, in_cluster_new):
    prev_clusters = parse_cluster_file(in_cluster_old)
    new_clusters = parse_cluster_file(in_cluster_new)
    score_data = parse_score_data(in_score)

    ok, nok = 0, 0
    all_pos, all_neg = 0, 0

    for pair in prev_clusters.keys():
        if pair in new_clusters.keys():

            prev_cluster = prev_clusters[pair]
            new_cluster = new_clusters[pair]
            is_correct = score_data[pair]

            if is_correct[:2] == "DA":
                all_pos += 1
            else:
                all_neg += 1

            prev_sense_w1, prev_sense_w2 = prev_cluster["w1_sense"], prev_cluster["w2_sense"]
            prev_sense_desc_w1, prev_sense_desc_w2 = prev_cluster["w1_senses"][prev_sense_w1], prev_cluster["w2_senses"][prev_sense_w2]
            new_sense_w1, new_sense_w2 = new_cluster["w1_sense"], new_cluster["w2_sense"]
            new_sense_desc_w1, new_sense_desc_w2 = new_cluster["w1_senses"][new_sense_w1], new_cluster["w2_senses"][new_sense_w2]

            if prev_sense_desc_w1 == new_sense_desc_w1 and prev_sense_desc_w2 == new_sense_desc_w2:
                if is_correct[:2] == "DA":
                    ok += 1
                else:
                    nok += 1
            else:
                print(pair)
                print(prev_sense_desc_w1, "|", prev_sense_desc_w2)
                print(new_sense_desc_w1, "|", new_sense_desc_w2)
                print()

    print(f"ok: {ok}, nok: {nok}")
    print(f"all ok: {all_pos}, all nok: {all_neg}")
    print(len(new_clusters.keys()))
    print(len(prev_clusters.keys()))

def parse_score_data(filename):
    data = {}
    with open(filename, "r", encoding="utf8") as f:
        for line in f:
            clean_data = re.sub("\t+", "\t", line.strip())
            w1, w2, score = clean_data.split("\t")
            data[(w1, w2)] = score

    return data

def parse_cluster_file(filename):
    clusters = {}
    w1, w2 = None, None

    with open(filename, "r", encoding="utf8") as f:
        for line in f:
            split_line = line.strip().split(" ")

            if line.startswith("Data for"):
                data = {}

                syn_ant, w1, w2 = split_line[2], split_line[4], split_line[6]
                data["synonym"] = syn_ant == "synonym"
                data["w1_senses"], data["w2_senses"] = {}, {}

            elif line.startswith(f"Sense data for"):
                w = split_line[3].strip(":\n \t")
                if w == w1:
                    w1_senses_bool = True
                elif w == w2:
                    w1_senses_bool = False
                else:
                    print(f"Error: Sense data for some other word: {w} instead of {w1}, {w2}")

            elif line.startswith("Predicted sense pair"):
                score = line.split(":")[-1].strip()
                data["score"] = score

            elif line == "\n":
                if w1 and w2:
                    clusters[(w1, w2)] = data

            else:

                if line.startswith("\t"):
                    num, sense = line.strip().split("\t")
                    if w1_senses_bool:

                        data["w1_senses"][num] = sense
                    else:
                        num, sense = line.strip().split("\t")
                        data["w2_senses"][num] = sense

                elif "w1_sense" not in data.keys() and line.startswith(w1):
                    data["w1_sense"] = line.split("(")[1].split(")")[0]

                elif line.startswith(w2):
                    data["w2_sense"] = line.split("(")[1].split(")")[0]

                elif line.strip().lower().replace(" ", '') == "sense\tdescription":
                    continue

                else:
                    print("Error: unknown line format:", line, w1, w2)

    return clusters
