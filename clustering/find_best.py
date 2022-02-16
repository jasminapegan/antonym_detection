from itertools import product

import file_helpers
from clustering.clusters import get_clusters_by_word, get_word_data
from scoring import get_avg_scores, score_clustering


def find_best_params_main(data_file, words_json_file, validation_file, out_dir, missing=[], use_algorithms=['kmeans']):
    r = 42
    n_init = [25, 50, 75, 100]
    min_samples = 3

    if 'kmeans' in use_algorithms:
        print("starting kmeans ...")

        algo = ['full', 'elkan']

        knn = [
            {'clusterer': None,
             'algorithm': 'kmeans',
             'parameters': {'random_state': 42,
                            'algorithm': 'full',
                            'n_init': n,
                            'algorithm': a},
             'out_file': None} for n, a in product(n_init, algo)
        ]

        find_best_clustering(data_file, words_json_file, validation_file, "%s/KNN.txt" % out_dir, knn, missing=missing)
        print("""
        *****************************\n
        *******kmeans FINISHED*******\n
        *****************************\n
        """)

    if 'spectral' in use_algorithms:
        print("starting spectral ...")

        affinity = ['cosine', 'euclidean', 'nearest_neighbors']
        gamma = [0.001, 0.01, 0.1, 1, 10]
        ks = [1, 2, 5, 10]

        spectral = [{'clusterer': None,
                     'algorithm': 'spectral',
                     'affinity': a,
                     'parameters': {'random_state': r,
                                    'n_neighbors': min_samples},
                     'out_file': "test"} for a in affinity]

        spectral += [{'clusterer': None,
                      'algorithm': 'spectral',
                      'affinity': 'precomputed',
                      'distance' : 'relative_cosine',
                      'k': k,
                      'parameters': {'random_state': r,
                                     'n_neighbors': min_samples},
                      'out_file': None} for k in ks]

        spectral += [{'clusterer': None,
                      'algorithm': 'spectral',
                      'affinity': 'precomputed',
                      'distance' : 'cosine',
                      'parameters': {'random_state': r,
                                     'n_neighbors': min_samples},
                      'out_file': None}]

        spectral += [{'clusterer': None,
                      'algorithm': 'spectral',
                      'affinity': 'rbf',
                      'gamma': g,
                      'parameters': {'random_state': r,
                                     'n_neighbors': min_samples},
                      'out_file': None} for g in gamma]

        find_best_clustering(data_file, words_json_file, validation_file, "%s/spectral.txt" % out_dir, spectral, missing=missing)
        print("""
        *****************************\n
        *******spectral FINISHED*****\n
        *****************************\n
        """)

    if 'agglomerative' in use_algorithms:
        print("starting agglomerative ...")

        affinity = ['cosine', 'euclidean', 'nearest_neighbors']
        linkage = ['complete', 'average']
        ks = [1, 2, 5, 10]

        agglomerative = [{'clusterer': None,
                          'algorithm': 'agglomerative',
                          'affinity': a,
                          'parameters': {'random_state': r,
                                         'linkage': l},
                          'out_file': None} for a, l in product(affinity, linkage)]

        agglomerative += [{'clusterer': None,
                           'algorithm': 'agglomerative',
                           'affinity': 'euclidean',
                           'parameters': {'random_state': r,
                                          'linkage': 'ward'},
                           'out_file': None}]

        agglomerative += [{'clusterer': None,
                           'algorithm': 'agglomerative',
                           'affinity': 'precomputed',
                           'k': k,
                           'parameters': {'random_state': r},
                           'out_file': None} for k in ks]

        find_best_clustering(data_file, words_json_file, validation_file, "%s/agglomerative.txt" % out_dir, agglomerative, missing=missing)
        print("""
        *****************************\n
        ***agglomerative FINISHED****\n
        *****************************\n
        """)
        if 'dbscan' in use_algorithms:
            print("starting dbscan ...")

            eps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

            dbscan = [{'clusterer': None,
                       'algorithm': 'dbscan',
                       'parameters': {'random_state': r,
                                      'metric': 'cosine',
                                      'eps': e,
                                      'min_samples': min_samples},
                       'out_file': None} for e in eps]

            find_best_clustering(data_file, words_json_file, validation_file, "%s/dbscan.txt" % out_dir,
                                 dbscan, missing=missing)
            print("""
            *****************************\n
            *******DBSCAN FINISHED*******\n
            *****************************\n
            """)


def find_best_clustering(data_file, words_json_file, validation_file, out_file, algorithm_params_list, missing=[]):
    words_json = file_helpers.load_json_word_data(words_json_file)
    val_data = file_helpers.load_validation_file_grouped(validation_file, embeddings=False, sentence_idx=3)
    algorithm_scores = {get_algorithm_id(algo): {} for algo in algorithm_params_list}
    idx = 0
    max_idx = file_helpers.file_len(data_file)

    while idx < max_idx:
        word, n_samples, n, words, sentences, embeddings, idx = get_word_data(data_file, words_json, idx)

        print("clustering %s ... %d / %d" % (word, idx, max_idx))

        if word == "iztočnica":
            continue
        elif word in missing or word not in val_data.keys():
            continue
        elif n < 2:
            continue

        word_val_data = val_data[word]
        val_ids = [sentences.index(s) for s in word_val_data['sentences'] if s.strip() in sentences]

        for params in algorithm_params_list:

            labels, silhouette_score, clusterer = get_clusters_by_word(word, sentences, embeddings, n_samples, n, params)
            params['clusterer'] = clusterer

            if labels is None:
                continue

            predicted_labels = [labels[i] for i in val_ids]

            if (len(predicted_labels) != len(word_val_data['labels'])):
                print("Labels not matching in length!")

            if (len(word_val_data['labels']) > 1):
                rand_score, adj_rand_score, completeness_score, f1_score, labels, confusion_matrix =\
                    score_clustering(predicted_labels, word_val_data['labels'])

                print("f1 score: %f" % f1_score)
                print(confusion_matrix)

                algorithm_scores[get_algorithm_id(params)][word] = {
                    'silhouette': silhouette_score,
                    'rand': rand_score,
                    'adjusted_rand': adj_rand_score,
                    'completeness': completeness_score,
                    'f1_score': f1_score,
                    'n_samples': n_samples,
                    'n_non_null': len(predicted_labels)
                }

    print(algorithm_scores)
    write_results(out_file, algorithm_scores, algorithm_params_list)


def write_results(out_file, algorithm_scores, algorithm_params_list):
    with open(out_file, "w", encoding="utf8") as f:
        for algo in algorithm_params_list:

            algo_id = get_algorithm_id(algo)
            algo_score = algorithm_scores[algo_id]
            algo['clusterer'] = None

            f.write("Score for %s, %s\n" % (algo_id, str(algo)))

            for word in algo_score.keys():
                f.write("\t'%s': %s\n" % (word, str(algo_score[word])))

            avg_scores = get_avg_scores(algo_score, ['silhouette', 'rand', 'adjusted_rand', 'completeness', 'f1_score'])
            f.write("Avg score: %s\n\n" % str(avg_scores))


def get_algorithm_id(algo):
    if 'linkage' not in algo.keys() or algo['linkage'] is None:
        return algo['algorithm']
    return " - ".join([algo['algorithm'], algo['linkage']])
