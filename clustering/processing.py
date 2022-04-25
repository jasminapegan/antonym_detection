import os
from statistics import mean

import matplotlib.pyplot as plt
import matplotlib.patches as mpatch

import file_helpers
from clustering.scoring import unsupervised_cluster_score


def get_results(data_file: str):
    results = []

    with open(data_file, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue

            data = line.strip().split("\t")
            word = data[0]
            score = float(data[2])
            results.append([word, score])

    with open(os.path.join(os.path.dirname(data_file), "clusters_eval.tsv"), "w", encoding="utf8") as outf:
        correct = [x for x in results if x[1] == 1.0]
        bad = [x for x in results if x[1] <= 0.0]
        outf.write("Correctly grouped: %d - %s\n" % (len(correct), ",".join([x[0] for x in correct])))
        outf.write("Badly grouped: %d - %s\n" % (len(bad), ",".join([x[0] for x in bad])))

        close = [[x for x in results if x[1] > 0.1 * i] for i in range(0, 9)]
        for i, c in enumerate(close):
            outf.write("Score %f or higher: %d - %s\n" % (0.1 * i, len(c), ",".join([x[0] for x in c])))

        loss = results
        loss.sort(key=lambda x: -x[1])
        outf.writelines(["\t".join([str(y) for y in x]) + "\n" for x in loss])

    return correct, close[5], close[0], bad


def find_best_clusters(file_list):
    correct, close, ok, bad = [], [], [], []

    for file in file_list:
        x, y, z, w = get_results(file)

        correct.append(set([xx[0] for xx in x]))
        close.append(set([yy[0] for yy in y]))
        ok.append(set([zz[0] for zz in z]))
        bad.append(set([ww[0] for ww in w]))

    correct_common = set.intersection(*correct)
    close_common = set.intersection(*close)
    ok_common = set.intersection(*ok)
    bad_common = set.intersection(*bad)

    correct_two = set.intersection(correct[0], correct[1]).union(
        set.intersection(correct[0], correct[2]).union(
            set.intersection(correct[1], correct[2])
        )
    )
    close_two = set.intersection(close[0], close[1]).union(
        set.intersection(close[0], close[2]).union(
            set.intersection(close[1], close[2])
        )
    )

    print("#Correct - all common: %d words: %s" % (len(correct_common), ", ".join(correct_common)))
    print("#Close - all common: %d words: %s" % (len(close_common), ", ".join(close_common)))
    print("#OK (>0) - all common: %d words: %s" % (len(ok_common), ", ".join(ok_common)))
    print("#Bad (<=0) - all common: %d words: %s" % (len(bad_common), ", ".join(bad_common)))

    print("#words: %s" % ", ".join([str(len(x) + len(y)) for x, y in zip(ok, bad)]))

    print("#Correct - two common: %d words: %s" % (len(correct_two), ", ".join(correct_two)))
    print("#Close - two common: %d words: %s" % (len(close_two), ", ".join(close_two)))

def compare_clusters(words, cluster_files, clustering_names, validation_data, validation_words, out_file, filename, sep="|"):
    if isinstance(words, str):
        words = file_helpers.get_unique_words(words)
    else:
        words = words

    print("loading data ...")
    val_words = file_helpers.count_words(validation_words, sep=sep)
    val_data = file_helpers.load_validation_file_grouped(validation_data)
    words = [w for w in words if w in val_words.keys() and w in val_data.keys() and val_words[w] > 1]

    cluster_data = []
    for file in cluster_files:
        print("Filtering file %s ..." % file)
        new_file = os.path.join(os.path.dirname(file), filename)
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
                f.write("\t".join(predicted + [val_data[word]['labels'][i], sentence]) + "\n")

def write_stats(score_files, data_files, out_file_stats, plot_data_file, skip_header=True):
    all_scores = {}
    for data_file in data_files:
        print("reading file: %s" % data_file)

        directory = os.path.dirname(data_file)
        filepath_stats = os.path.join(directory, out_file_stats)

        with open(filepath_stats, "w", encoding="utf8") as fs:
            fs.write("word\tsilhouette\tdb_score\tch_score\tdata_moments\tavg_moments\n")

            scores = {}
            data = file_helpers.load_result_file_grouped(data_file, embeddings=True, skip_header=skip_header)

            for word, word_data in data.items():
                print(word)
                embeddings = file_helpers.convert_to_np_array(word_data['embeddings'])
                labels = word_data['labels']

                score = unsupervised_cluster_score(embeddings, labels)
                fs.writelines("\t".join([str(x) for x in score]) + "\n")
                scores[word] = score

                if word == "Anglija":
                    break

        all_scores[data_file] = scores

    with open(out_file_stats, "w", encoding="utf8") as f:
        print("processing scores")
        all_processed = process_scores(all_scores)
        f.write("algorithm\tsilhouette\tdb_score\tch_score\tdata_moments\tavg_moments\n")

        for word, value in all_processed.items():
            f.write("*** %s ***\n" % word)
            f.writelines("\n".join(["\t".join([alg] + [str(x) for x in line]) for alg, line in value.items()]))

    save_correlation_data(score_files, all_processed, plot_data_file)

def process_scores(scores_by_algo):
    algos = list(scores_by_algo.keys())
    words = list(scores_by_algo[algos[0]].keys())
    processed = {}

    for word in words:
        processed[word] = {}

        for algo in algos:
            algo_name = algo.split('/')[1].split("-")[0]
            algo_score = scores_by_algo[algo][word]
            processed[word][algo_name] = algo_score

    return processed

def save_correlation_data(score_files, stats, out_file):
    scores_dict = {}
    for score_file in score_files:
        algo = score_file.split('/')[1].split('-')[0]
        scores_dict[algo] = {}

        scores = file_helpers.load_file(score_file, skip_header=True)
        for line in scores:
            scores_dict[algo][line[0]] = line[2]

    words = [w for w in stats.keys() if w in scores_dict[algo].keys()]
    print([w for w in stats.keys() if w not in scores_dict[algo].keys()])
    moment_names = ["mean", "variance", "stddev", "skewness", "kurtosis", "6th", "7th", "8th"]

    with open(out_file, "w", encoding="utf8") as f:
        for algo in scores_dict.keys():
            f.write("Algorithm: " + algo + "\n")

            algo_stats = [stats[w][algo] for w in words]
            algo_scores = scores_dict[algo]

            score_data = [float(algo_scores[w]) for w in words]
            score_moment_names = ["silhouette", "db_score", "ch_score"] +\
                          ["data_moments-" + m for m in moment_names] + ["avg_moments-" + m for m in moment_names]
            score_names = ["silhouette", "db_score", "ch_score", "data_moments", "avg_moments"]

            f.write("Score names: " + " ".join(score_moment_names) + "\n")

            for i, score_name in enumerate(score_names):

                if "moments" in score_name:
                    moment_stats = [algo_stats[w][i] for w in range(len(words))]
                    for moment in moment_names:
                        f.write("Score: " + score_name + "-" + moment + "\n")
                        stats_data = [moment_stats[w][moment] for w in range(len(words))]
                        write_stats_score_data(f, stats_data, score_data)
                else:
                    f.write("Score: " + score_name + "\n")
                    stats_data = [algo_stats[w][i] for w in range(len(words))]
                    write_stats_score_data(f, stats_data, score_data)

        f.write("Algorithm: all\n")
        algos = list(scores_dict.keys())
        algo_stats = [mean([stats[w][algo][i] for algo in algos]) for w in words]
        algo_scores = [mean([scores_dict[algo][w] for algo in algos]) for w in words]

        score_data = [float(algo_scores[w]) for w in words]
        score_moment_names = ["silhouette", "db_score", "ch_score"] + \
                             ["data_moments-" + m for m in moment_names] + ["avg_moments-" + m for m in moment_names]
        score_names = ["silhouette", "db_score", "ch_score", "data_moments", "avg_moments"]

        f.write("Score names: " + " ".join(score_moment_names) + "\n")

        for i, score_name in enumerate(score_names):

            if "moments" in score_name:
                moment_stats = [algo_stats[w][i] for w in range(len(words))]
                for moment in moment_names:
                    f.write("Score: " + score_name + "-" + moment + "\n")
                    stats_data = [moment_stats[w][moment] for w in range(len(words))]
                    write_stats_score_data(f, stats_data, score_data)
            else:
                f.write("Score: " + score_name + "\n")
                stats_data = [algo_stats[w][i] for w in range(len(words))]
                write_stats_score_data(f, stats_data, score_data)


def write_stats_score_data(f, stats, scores):
    f.write(" ".join([str(x) for x in stats]) + "\t" + " ".join([str(x) for x in scores]) + "\n")


def write_centroids(data_files, out_file):
    for data_file in data_files:

        directory = os.path.dirname(data_file)
        filepath_centroids = os.path.join(directory, out_file)

        with open(filepath_centroids, "w", encoding="utf8") as fc:
            fc.write("word\tlabel\tcentroid\n")

            data = file_helpers.load_result_file_grouped(data_file, embeddings=True)

            for word, word_data in data.items():
                embeddings = file_helpers.convert_to_np_array(word_data['embeddings'])
                labels = word_data['labels']
                centroids = get_centroids(embeddings, labels)

                fc.writelines("\n".join(["\t".join(
                    [word, label, " ".join([str(x) for x in centroids[label]])]
                ) for label in set(labels)]))

def get_centroids(embeddings, labels):
    clusters = {c: [] for c in labels}
    for c, x in zip(labels, embeddings):
        clusters[c].append(x)

    centroids = {c: get_centroid(x) for c, x in clusters.items()}
    return centroids

def get_centroid(arr):
    return [sum(m) / len(m) for m in zip(*arr)]



def prepare_data(in_files, out_file, labels_file, words=None):
    # prepare data for online visualization
    for in_file in in_files:
        directory = os.path.dirname(in_file)

        with open(in_file, "r", encoding="utf8") as f:
            out_file_path = os.path.join(directory, out_file)

            with open(out_file_path, "w", encoding="utf8") as out_f:
                labels_file_path = os.path.join(directory, labels_file)

                with open(labels_file_path, "w", encoding="utf8") as labels_f:
                    labels_f.write("label\tword\tsentence\n")

                    for line in f:
                        label, word, sentence, embedding = line.strip().split("\t")

                        if not words or word in words:
                            out_f.write(embedding.replace(" ", "\t") + "\n")
                            labels_f.write("\t".join([label, word, sentence]) + "\n")


result_files =  ["out_latest/agglomerative/agglomerative-affinity=precomputed,distance=relative_cosine,k=20_data.tsv",
                 "out_latest/kmeans/kmeans-algorithm=full,n_init=130_data.tsv",
                 "out_latest/spectral/spectral-affinity=cosine,n_neighbors=3_data.tsv"]

score_files =  ["out_latest/agglomerative/agglomerative-affinity=precomputed,distance=relative_cosine,k=20.tsv",
                "out_latest/kmeans/kmeans-algorithm=full,n_init=130.tsv",
                "out_latest/spectral/spectral-affinity=cosine,n_neighbors=3.tsv"]

#write_stats(score_files, result_files, "stats.tsv", "plot_data.txt")