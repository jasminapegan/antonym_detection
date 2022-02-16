""" Version 2 treats number of clusters as an unknown parameter. """
from typing import List, Dict

from clustering import algorithms
from data import grouped_file

import file_helpers
from clustering.clusters_v2 import get_clusters_by_word_v2
from data.embeddings import WordEmbeddings
from clustering.scoring import get_avg_scores


def find_best_params_main_v2(data_file: str,
                             words_json_file: str,
                             validation_file: str,
                             out_dir: str,
                             use_algorithms: List[str]) -> None:
    n_init = [25, 50, 75, 100]
    min_samples = [3]

    if 'kmeans' in use_algorithms:
        algo = ['full', 'elkan']

        kmeans = algorithms.KMeansAlgorithm.get_clusterer_list(algorithms=algo, n_inits=n_init)

        find_best_clustering_v2(data_file, words_json_file, validation_file, "%s/KNN.txt" % out_dir, kmeans)

    if 'spectral' in use_algorithms:
        affinity = ['cosine', 'euclidean', 'nearest_neighbors']
        distance = ['cosine', 'relative_cosine']
        gamma = [0.001, 0.01, 0.1, 1, 10]
        ks = [1, 2, 5, 10]

        spectral = algorithms.SpectralAlgorithm.get_clusterer_list(
            affinity=affinity, n_neighbors=min_samples, distance=distance, ks=ks, gamma=gamma)

        find_best_clustering_v2(data_file, words_json_file, validation_file, "%s/spectral.txt" % out_dir, spectral)

    if 'agglomerative' in use_algorithms:
        affinity = ['cosine', 'euclidean', 'nearest_neighbors']
        linkage = ['complete', 'average', 'ward']
        ks = [1, 2, 5, 10]

        agglomerative = algorithms.AgglomerativeAlgorithm.get_clusterer_list(affinity=affinity, linkage=linkage, ks=ks)

        find_best_clustering_v2(data_file, words_json_file, validation_file, "%s/agglomerative.txt" % out_dir, agglomerative)

        if 'dbscan' in use_algorithms:
            eps = [0.1, 0.3, 0.5, 0.7, 0.9]
            leaf_size = [1, 3, 5, 7]

            dbscan = algorithms.DbscanAlgorithm.get_clusterer_list(eps=eps, min_samples=min_samples, leaf_size=leaf_size)

            find_best_clustering_v2(data_file, words_json_file, validation_file, "%s/dbscan.txt" % out_dir, dbscan)


def find_best_kmeans(data_file: str, words_json_file: str, validation_file: str, out_dir: str) -> None:
    algo = ['full']#, 'elkan']
    n_init = [25] #, 50, 75, 100]
    kmeans = algorithms.KMeansAlgorithm.get_clusterer_list(algorithms=algo, n_inits=n_init)
    find_best_clustering_v2(data_file, words_json_file, validation_file, out_dir, kmeans)


def find_best_clustering(data_file: str,
                         words_json_file: str,
                         validation_file: str,
                         out_dir: str,
                         algorithm_list: List[algorithms.ClusteringAlgorithm]) -> None:

    print("*****\nSTARTING %s" % str([a.get_algorithm_id() for a in algorithm_list]))

    best_clustering = BestClustering(words_json_file, validation_file, data_file)
    best_clustering.find_best(algorithm_list, out_dir)

def find_best_clustering_v2(data_file: str,
                            words_json_file: str,
                            validation_file: str,
                            out_dir: str,
                            algorithm_list: List[algorithms.ClusteringAlgorithm]) -> None:

    print("*****\nSTARTING %s" % str([a.get_algorithm_id() for a in algorithm_list]))

    best_clustering = BestClustering(words_json_file, validation_file, data_file)
    best_clustering.find_best(algorithm_list, out_dir)


class BestClustering:

    def __init__(self, words_json_file: str, validation_file: str, data_file: str):
        self.words_json = file_helpers.load_json_word_data(words_json_file)
        self.val_data = file_helpers.load_validation_file_grouped(validation_file, embeddings=False, indices=True, sentence_idx=3)
        self.word_data_generator = grouped_file.word_data_gen(data_file)
        self.word_embeddings = WordEmbeddings()


    def find_best(self, algorithm_list: List[algorithms.ClusteringAlgorithm], out_dir: str) -> None:
        for word_data in self.word_data_generator:

            word = word_data.words[0]
            n = len(self.words_json[word])

            if word not in self.val_data.keys():
                continue

            word_data = self.prepare_word_data(word, word_data)

            for algorithm_data in algorithm_list:

                self.execute_clustering(algorithm_data, word_data, out_dir, n)

                """labels, silhouette_score = get_clusters_by_word_v2(word_data, n, algorithm_data, out_dir)

                #if labels is None:
                #    continue

                word_data.set_predicted_labels(labels)

                if (len(word_data.validation_labels) > 1):
                    algorithm_data.score_method(word_data, silhouette_score)"""

            algorithm_scores = {a.get_algorithm_id(): a.score for a in algorithm_list}
            print(algorithm_scores)

        self.write_results(out_dir + "results_all.txt", algorithm_scores, algorithm_list)
        print("\n*****FINISHED*****")


    @staticmethod
    def execute_clustering(algorithm_data, word_data, out_dir, n=None):
        if n:
            ns = [n]
        else:
            ns = range(10)

        for n_clusters in ns:
            labels, silhouette_score = get_clusters_by_word_v2(word_data, n_clusters, algorithm_data, out_dir)

            if labels is not None:
                word_data.set_predicted_labels(labels)

                if (len(word_data.validation_labels) > 1):
                    algorithm_data.score_method(word_data, silhouette_score)


    def prepare_word_data(self, word: str, word_data: grouped_file.WordData):
        word_val_data = self.val_data[word]

        # add missing sentences + embeddings
        missing_sentences = [s for s in word_val_data['sentences'] if s not in word_data.sentences]

        if missing_sentences:
            word_data.add_missing_sentences(missing_sentences, self.word_embeddings, word_val_data)

        word_data.val_ids = [word_data.sentences.index(s) for s in word_val_data['sentences'] if s in word_data.sentences]
        word_data.validation_labels = word_val_data['labels']

        return word_data

    @staticmethod
    def write_results(out_file: str,
                      algorithm_scores: Dict,
                      algorithm_params_list: List[algorithms.ClusteringAlgorithm]) -> None:

        with open(out_file, "a", encoding="utf8") as f:
            for algo in algorithm_params_list:

                algo_id = algo.get_algorithm_id()
                algo_score = algorithm_scores[algo_id]

                f.write("Score for %s, %s\n" % (algo_id, str(algo)))

                for word in algo_score.keys():
                    f.write("\t'%s': %s\n" % (word, str(algo_score[word])))

                avg_scores = get_avg_scores(algo_score,
                                            ['silhouette', 'rand', 'adjusted_rand', 'completeness', 'f1_score'])
                f.write("Avg score: %s\n\n" % str(avg_scores))
