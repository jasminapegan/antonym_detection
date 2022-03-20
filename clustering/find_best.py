""" Version 2 treats number of clusters as an unknown parameter. """
import os
import warnings

import file_helpers
from data import word
from clustering import algorithms
from clustering.scoring import get_avg_scores
from data.embeddings import WordEmbeddings
from typing import List, Dict
from sklearn import metrics


def find_best_params_main(data_file: str, words_file: str, validation_file: str, out_dir: str,
                             use_algorithms: List[str]):
    """
    TODO: split into smaller functions

    :param data_file:
    :param words_file:
    :param validation_file:
    :param out_dir:
    :param use_algorithms:
    :return:
    """

    n_init = [25, 50, 75, 100]
    min_samples = [3]

    if 'kmeans' in use_algorithms:
        algo = ['full', 'elkan']

        kmeans = algorithms.KMeansAlgorithm.get_clusterer_list(algorithms=algo, n_inits=n_init)

        find_best_clustering(data_file, words_file, validation_file, "%s/KNN.txt" % out_dir, kmeans)

    if 'spectral' in use_algorithms:
        affinity = ['cosine', 'euclidean', 'nearest_neighbors']
        distance = ['cosine', 'relative_cosine']
        gamma = [0.001, 0.01, 0.1, 1, 10]
        ks = [1, 2, 5, 10]

        spectral = algorithms.SpectralAlgorithm.get_clusterer_list(
            affinity=affinity, n_neighbors=min_samples, distance=distance, ks=ks, gamma=gamma)

        find_best_clustering(data_file, words_file, validation_file, "%s/spectral.txt" % out_dir, spectral)

    if 'agglomerative' in use_algorithms:
        affinity = ['cosine', 'euclidean', 'nearest_neighbors']
        linkage = ['complete', 'average', 'ward']
        ks = [1, 2, 5, 10]

        agglomerative = algorithms.AgglomerativeAlgorithm.get_clusterer_list(affinity=affinity, linkage=linkage, ks=ks)

        find_best_clustering(data_file, words_file, validation_file, "%s/agglomerative.txt" % out_dir, agglomerative)

        if 'dbscan' in use_algorithms:
            eps = [0.1, 0.3, 0.5, 0.7, 0.9]
            leaf_size = [1, 3, 5, 7]

            dbscan = algorithms.DbscanAlgorithm.get_clusterer_list(eps=eps, min_samples=min_samples, leaf_size=leaf_size)

            find_best_clustering(data_file, words_file, validation_file, "%s/dbscan.txt" % out_dir, dbscan)

def find_best_kmeans(data_file: str, words_file: str, validation_file: str, out_dir: str='out'):
    """
    Execute search for best kmeans algorithm over predefined parameters. Score word embedding clusterings and write out
    scores for each algorithm.

    :param data_file: input embeddings file
    :param words_file: words data (used to determine number of senses)
    :param validation_file: labeled sentences data
    :param out_dir: output directory for separate algorithm score files
    :return: None
    """

    algo = ['full', 'elkan']
    n_init = [20 * i for i in range(1, 5)]
    kmeans = algorithms.KMeansAlgorithm.get_clusterer_list(algorithms=algo, n_inits=n_init)
    find_best_clustering(data_file, words_file, validation_file, out_dir, kmeans)

def find_best_clustering(data_file: str,
                         words_file: str,
                         validation_file: str,
                         out_dir: str,
                         algorithm_list: List[algorithms.ClusteringAlgorithm]) -> None:

    print("*****\nSTARTING %s" % str([a.id for a in algorithm_list]))

    best_clustering = BestClustering(words_file, validation_file, data_file)
    best_clustering.find_best(algorithm_list, out_dir)

class BestClustering:

    Scores = Dict[str, Dict[str, float]]

    def __init__(self, words_file: str, validation_file: str, data_file: str, out_dir: str='out'):
        self.words_json = file_helpers.words_data_to_dict(words_file, header=False, skip_num=True)
        self.val_data = file_helpers.load_validation_file_grouped(validation_file, indices=True)
        self.word_data_generator = word.word_data_gen(data_file, progress=500)
        self.word_embeddings = WordEmbeddings()
        self.scores = {}
        self.out_dir = out_dir

    @staticmethod
    def get_clusters_by_word(word_data: word.WordData, n_clusters: int, algorithm: algorithms.ClusteringAlgorithm) \
            -> (List[int], float):
        """
        Execute clustering with given algorithm on given word data.

        :param word_data: word data including sentences and embeddings
        :param n_clusters: number of clusters to use with algorithm
        :param algorithm: clustering algorithm to use
        :return: list of labels and silhouette score
        """

        if word_data.n_sentences < n_clusters:
            # print("Too little samples for %s: %d samples, %d clusters." % (word_data.word, n_samples, n))
            return None, None

        else:
            labels = algorithm.predict(word_data.embeddings, n_clusters)

            """with open(algorithm.out_file, "a", encoding="utf8") as outf:
                # out_data = list(zip(labels, [word] * n_samples, sentences)) #, embeddings))
                # file_helpers.write_grouped_data(outf, sorted(out_data, key=lambda x: x[0])) #, centroids=clusterer.cluster_centers_)

                str_embeddings = [" ".join([str(x) for x in e]) for e in word_data.embeddings]
                out_data = list(zip([str(x) for x in labels],
                                    [word_data.word] * word_data.n_sentences,
                                    word_data.sentences))#,
                                    #str_embeddings))
                outf.writelines(["\t".join(line) + "\n" for line in out_data])"""

            silhouette = None

            try:
                silhouette = metrics.silhouette_score(word_data.embeddings, labels, metric='cosine')
                # print("silhouette score: %f" % silhouette)
            except Exception as e:
                pass
                #print("silhouette score could not be calculated: %s" % str(e))

            return labels, silhouette

    def find_best(self, algorithm_list: List[algorithms.ClusteringAlgorithm], out_dir: str):
        """
        Find best clustering algorithm from given list of algorithms, output scores into out_dir.

        :param algorithm_list: a list of algorithms to use
        :param out_dir: directory where scores are saved
        :return: None
        """
        algorithm_scores = []

        for word_data in self.word_data_generator:

            word = word_data.words[0]
            n = len(self.words_json[word])

            if word not in self.val_data.keys():
                continue

            word_data = self.prepare_word_data(word, word_data)

            for algorithm_data in algorithm_list:
                self.execute_clustering(algorithm_data, word_data, n=n)
                #algorithm_scores.append(algorithm_data.score)

        #print(algorithm_scores)
        #self.write_results(os.path.join(out_dir, "results_all.txt"), algorithm_list)
        print("\n*****FINISHED*****")

    def execute_clustering(self, algorithm_data: algorithms.ClusteringAlgorithm, word_data: word.WordData, n: int=None):
        """
        Execute clustering with given 'algorithm_data' on given 'word_data', print results into file in 'out_dir'.

        :param algorithm_data: algorithm to be used with values for parameters
        :param word_data: WordData representing data on selected word
        :param n: number of clusters. If None, all the values from 1 to 10 are tested. (default None)
        :return: None
        """

        """if n:
            ns = [n]
        else:
            ns = range(10)"""

        #for n_clusters in ns:
        labels, silhouette_score = self.get_clusters_by_word(word_data, n, algorithm_data)

        if labels is not None:
            word_data.set_predicted_labels(labels)

            if (len(word_data.validation_labels) > 1):
                algorithm_data.score_method(word_data, silhouette_score, n)


        #if not n:
        #    algorithm_data.get_best_n_clusters(word_data)

    def prepare_word_data(self, word: str, word_data: word.WordData) -> word.WordData:
        """
        Check if validation data contains sentences not included in WordData. For each sentence, calculate the observed
        word's embedding.

        :param word: the observed word
        :param word_data: already processed word data
        :return: updated WordData object
        """

        word_val_data = self.val_data[word]

        # add missing sentences + embeddings
        missing_sentences = [s for s in word_val_data['sentences'] if s not in word_data.sentences]
        if missing_sentences:
            word_data.add_missing_sentences(missing_sentences, self.word_embeddings, word_val_data)

        word_data.val_ids = [word_data.sentences.index(s) for s in word_val_data['sentences']]# if s in word_data.sentences]
        word_data.validation_labels = word_val_data['labels']

        return word_data

    @staticmethod
    def write_results(out_file: str, algorithm_params_list: List[algorithms.ClusteringAlgorithm]):
        """
        Write scores of given algorithms to 'out_file'.

        :param out_file: file to write scores to
        :param algorithm_scores: a dict of scores (dict of float per word) per algorithm
        :param algorithm_params_list:
        :return: None
        """
        if os.path.exists(out_file):
            os.remove(out_file)

        best_algo = {'score': 0.0, 'algo': None}

        with open(out_file, "a", encoding="utf8") as f:
            for algo in algorithm_params_list:

                f.write("Score for %s\n" % algo.id)

                for word in algo.score.keys():
                    f.write("\t'%s': %s\n" % (word, str(algo.score[word])))

                avg_scores = get_avg_scores(algo.score, ['adjusted_rand','completeness', 'f1_score', 'silhouette'])
                f.write("Avg score: %s\n\n" % str(avg_scores))

                algo_score = avg_scores['adjusted_rand'][0]
                if algo_score > best_algo['score']:

                    best_algo['score'] = algo_score
                    best_algo['algo'] = algo.id

            f.write("Best algorithm: %s score: %s" % (best_algo['algo'], best_algo['score']))