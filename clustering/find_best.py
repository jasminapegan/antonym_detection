import os

import file_helpers
from data import word
from clustering import algorithms
from clustering.scoring import get_avg_scores
from data.embeddings import WordEmbeddings
from typing import List, Dict, Union
from sklearn import metrics


def find_best_kmeans(data_file: str, words_file: str, validation_file: str, out_dir: str='out', output_vectors=False):
    """
    Execute search for best kmeans algorithm over predefined parameters. Score word embedding clusterings and write out
    scores for each algorithm.

    :param data_file: input embeddings file
    :param words_file: words data (used to determine number of senses)
    :param validation_file: labeled sentences data
    :param out_dir: output directory for separate algorithm score files
    :param output_vectors: if set to True, save results data
    :return: None
    """

    algo = ['full', 'elkan']
    n_init = [20]# * i for i in range(1, 5)]
    kmeans = algorithms.KMeansAlgorithm.get_clusterer_list(algorithms=algo, n_inits=n_init, out_dir=out_dir)
    find_best_clustering(data_file, words_file, validation_file, out_dir, kmeans, res_file="kmeans_all.txt",
                         output_vectors=output_vectors)

def find_best_spectral(data_file: str, words_file: str, validation_file: str, out_dir: str='out', output_vectors=False):
    """
    Execute search for best spectral algorithm over predefined parameters. Score word embedding clusterings and write
    scores for each algorithm.

    :param data_file: input embeddings file
    :param words_file: words data (used to determine number of senses)
    :param validation_file: labeled sentences data
    :param out_dir: output directory for separate algorithm score files
    :param output_vectors: if set to True, save results data
    :return: None
    """

    affinity = ['cosine', 'rbf', 'precomputed']
    distance = ['cosine', 'relative_cosine']
    gamma = [0.001, 0.01, 0.1, 1, 10, 100]
    ks = [5*i for i in range(1, 21)]
    min_samples = [3]

    spectral = algorithms.SpectralAlgorithm.get_clusterer_list(
        affinity=affinity, n_neighbors=min_samples, distance=distance, ks=ks, gamma=gamma, out_dir=out_dir)

    find_best_clustering(data_file, words_file, validation_file, out_dir, spectral, res_file="spectral_all.txt",
                         output_vectors=output_vectors)

def find_best_agglomerative(data_file: str, words_file: str, validation_file: str, out_dir: str='out',
                            output_vectors=False):
    """
    Execute search for best agglomerative algorithm over predefined parameters. Score word embedding clusterings and
    write scores for each algorithm.

    :param data_file: input embeddings file
    :param words_file: words data (used to determine number of senses)
    :param validation_file: labeled sentences data
    :param out_dir: output directory for separate algorithm score files
    :param output_vectors: if set to True, save results data
    :return: None
    """

    affinity = ['cosine', 'euclidean', 'l1', 'precomputed']
    linkage = ['complete', 'average', 'ward']
    distance = ['relative_cosine']
    ks = [5*i for i in range(1, 21)]

    agglomerative = algorithms.AgglomerativeAlgorithm.get_clusterer_list(affinity=affinity, linkage=linkage, ks=ks,
                                                                         distance=distance, out_dir=out_dir)

    find_best_clustering(data_file, words_file, validation_file, out_dir, agglomerative,
                         res_file="agglomerative_all.txt", output_vectors=output_vectors)

def find_best_dbscan(data_file: str, words_file: str, validation_file: str, out_dir: str = 'out', output_vectors=False):
    """
    Execute search for best dbscan algorithm over predefined parameters. Score word embedding clusterings and
    write scores for each algorithm.

    :param data_file: input embeddings file
    :param words_file: words data (used to determine number of senses)
    :param validation_file: labeled sentences data
    :param out_dir: output directory for separate algorithm score files
    :param output_vectors: if set to True, save results data
    :return: None
    """
    eps = [0.1 * i for i in range(5, 10)]
    leaf_size = [i for i in range(1, 10)]
    min_samples = [1, 2, 3]

    dbscan = algorithms.DbscanAlgorithm.get_clusterer_list(eps=eps, min_samples=min_samples,
                                                           leaf_size=leaf_size, out_dir=out_dir)

    find_best_clustering(data_file, words_file, validation_file, out_dir, dbscan,
                         res_file="dbscan_all.txt", output_vectors=output_vectors)

def find_best_all(data_file: str, words_file: str, validation_file: str, out_dir: str = 'best', output_vectors=True, use_pos=False):

    kmeans = algorithms.KMeansAlgorithm.get_clusterer_list(algorithms=['full'], n_inits=[130], out_dir=out_dir)

    spectral = algorithms.SpectralAlgorithm.get_clusterer_list(affinity=['cosine'], n_neighbors=[3],
                                                               distance=[], ks=[], gamma=[], out_dir=out_dir)

    agglomerative = algorithms.AgglomerativeAlgorithm.get_clusterer_list(affinity=['precomputed'], linkage=[], ks=[20],
                                                                         distance=['relative_cosine'], out_dir=out_dir)

    find_best_clustering(data_file, words_file, validation_file, out_dir, kmeans + spectral + agglomerative,
                         res_file="best_results.txt", output_vectors=output_vectors, use_pos=use_pos)

"""def ensemble_clustering(data_file: str, words_file: str, validation_file: str, out_dir: str = 'best', output_vectors=True):

    #kmeans = "out/kmeans/kmeans-algorithm=full,n_init=130_data.tsv"
    #spectral = "out/spectral/spectral-affinity=cosine,n_neighbors=3_data.tsv"
    #agglomerative = "out/agglomerative/agglomerative-affinity=precomputed,distance=relative_cosine,k=20_data.tsv"

    kmeans = algorithms.KMeansAlgorithm.get_clusterer_list(algorithms=['full'], n_inits=[130], out_dir=out_dir)

    spectral = algorithms.SpectralAlgorithm.get_clusterer_list(affinity=['cosine'], n_neighbors=[3],
                                                               distance=[], ks=[], gamma=[], out_dir=out_dir)

    agglomerative = algorithms.AgglomerativeAlgorithm.get_clusterer_list(affinity=['precomputed'], linkage=[], ks=[20],
                                                                         distance=['relative_cosine'], out_dir=out_dir)
    #ensemble = algorithms.EnsembleClustering.get_algorithm_results([kmeans, spectral, agglomerative])
    ensemble = algorithms.EnsembleClustering(out_dir=out_dir)
    ensemble.set_clusterer_list(kmeans + spectral + agglomerative)

    find_best_clustering(data_file, words_file, validation_file, out_dir, [ensemble],
                         res_file="best_results.txt", output_vectors=output_vectors)"""

def find_best_clustering(data_file: str,
                         words_file: str,
                         validation_file: str,
                         out_dir: str,
                         algorithm_list: List[algorithms.ClusteringAlgorithm],
                         res_file: str,
                         output_vectors: bool=True,
                         use_pos: bool=False) -> None:

    print("*****\nSTARTING %s" % str([a.id for a in algorithm_list]))

    #best_clustering = BestClustering(words_file, validation_file, data_file)
    #best_clustering.process_single_clusters(out_dir)

    best_clustering = BestClustering(words_file, validation_file, data_file, out_dir=out_dir, use_pos=use_pos)#, reduce_embeddings=False)
    best_clustering.find_best(algorithm_list, out_dir, output_vectors=output_vectors, res_file=res_file)

class BestClustering:

    Scores = Dict[str, Dict[str, float]]

    def __init__(self, words_file: str, validation_file: str, data_file: str, out_dir: str='out',
                 reduce_embeddings: bool=False, use_pos=False):
        if words_file:
            self.words_json = file_helpers.words_data_to_dict(words_file, header=False, skip_num=True)
        if validation_file:
            self.val_data = file_helpers.load_validation_file_grouped(validation_file, indices=True, use_pos=use_pos)
        if data_file:
            self.word_data_generator = word.word_data_gen(data_file, progress=5000)
        self.word_embeddings = WordEmbeddings()
        self.scores = {}
        self.out_dir = out_dir

        self.reduce_embeddings = reduce_embeddings

        #if reduce_embeddings:
        #    self.reducer = umap.UMAP()
        #    self.scaler = StandardScaler()

    @staticmethod
    def get_clusters_by_word(word_data: word.WordData, n_clusters: int, algorithm: algorithms.ClusteringAlgorithm,
                             output_vectors: bool=False) -> (List[int], float):
        """
        Execute clustering with given algorithm on given word data.

        :param word_data: word data including sentences and embeddings
        :param n_clusters: number of clusters to use with algorithm
        :param algorithm: clustering algorithm to use
        :param output_vectors: whether to save output data
        :return: list of labels and silhouette score
        """

        if word_data.n_sentences < n_clusters:
            # print("Too little samples for %s: %d samples, %d clusters." % (word_data.word, n_samples, n))
            return None, None

        else:
            labels = algorithm.predict(word_data.embeddings, n_clusters)

            if output_vectors:
                with open(algorithm.data_file, "a", encoding="utf8") as outf:
                    str_embeddings = [" ".join([str(x) for x in e]) for e in word_data.embeddings]
                    out_data = list(zip([str(x) for x in labels],
                                        [word_data.word] * word_data.n_sentences,
                                        word_data.sentences,
                                        str_embeddings))
                    outf.writelines(["\t".join(line) + "\n" for line in out_data])

            try:
                silhouette = metrics.silhouette_score(word_data.embeddings, labels, metric='cosine')
            except Exception:
                silhouette = None

            return labels, silhouette

    def find_best(self, algorithm_list: List[algorithms.ClusteringAlgorithm], out_dir: str, output_vectors: bool=False,
                  res_file: str="results_all.txt"):
        """
        Find best clustering algorithm from given list of algorithms, output scores into out_dir.

        :param algorithm_list: a list of algorithms to use
        :param out_dir: directory where scores are saved
        :param output_vectors: whether to save every algorithm's output
        :param res_file: where to save score data
        :return: None
        """
        skipped_words = []

        for word_data in self.word_data_generator:

            word = word_data.words[0]
            pos = word_data.pos
            n = len(self.words_json[word])

            print(word, pos, n)

            if n < 2:
                #print("skipping word", word)
                skipped_words.append(word)
                continue

            if word not in self.val_data.keys():
                #print(word)
                continue

            word_data = self.prepare_word_data(word, pos, word_data)#, reduce_embeddings=self.reduce_embeddings)

            if word_data is None:
                continue

            for algorithm_data in algorithm_list:
                self.execute_clustering(algorithm_data, word_data, n=n, output_vectors=output_vectors)

        self.write_results(os.path.join(out_dir, res_file), algorithm_list)

        print("%d skipped words: %s" % (len(skipped_words), ", ".join(skipped_words)))
        print("\n*****FINISHED*****")


    def process_single_clusters(self, out_dir: str):
        """
        Process single clusters, output scores into out_dir.

        :param out_dir: directory where scores are saved
        :return: None
        """
        out_file = os.path.join(out_dir, "single_cluster.txt")

        for word_data in self.word_data_generator:

            word = word_data.words[0]
            n = len(self.words_json[word])

            if n != 1:
                continue

            if word not in self.val_data.keys():
                print("missing word %s" % word)
                continue

            word_data = self.prepare_word_data(word, word_data)
            with open(out_file, "a", encoding="utf8") as outf:
                out_data = list(zip([word_data.word] * word_data.n_sentences, word_data.sentences))
                outf.writelines(["0\t" + "\t".join(line) + "\n" for line in out_data])

        print("\n*****FINISHED*****")

    def execute_clustering(self, algorithm_data: algorithms.ClusteringAlgorithm, word_data: word.WordData, n: int=None,
                           output_vectors=False):
        """
        Execute clustering with given 'algorithm_data' on given 'word_data', print results into file in 'out_dir'.

        :param algorithm_data: algorithm to be used with values for parameters
        :param word_data: WordData representing data on selected word
        :param n: number of clusters. If None, all the values from 1 to 10 are tested. (default None)
        :param output_vectors: whether to save output data and embeddings
        :return: None
        """

        labels, silhouette_score = self.get_clusters_by_word(word_data, n, algorithm_data, output_vectors=output_vectors)

        if labels is not None:
            word_data.set_predicted_labels(labels)

            #if (len(word_data.validation_labels) > 1):
            algorithm_data.score_method(word_data, silhouette_score, n)
            #else:
            #    print("Too little validation labels to score word %s" % word_data.word)

    def prepare_word_data(self, word: str, pos: str, word_data: word.WordData, reduce_embeddings=False) -> Union[word.WordData, None]:
        """
        Check if validation data contains sentences not included in WordData. For each sentence, calculate the observed
        word's embedding.

        :param word: the observed word
        :param word_data: already processed word data
        :return: updated WordData object
        """

        pos_tags = list(self.val_data[word].keys())
        pos_tag = pos
        if pos not in pos_tags and pos not in "SGPRZKDVLMO":
            if len(pos_tags) == 1:
                pos_tag = pos_tags[0]
            elif "X" in pos_tags:
                pos_tag = "X"
            elif "U" in pos_tags:
                pos_tag = "U"
            else:
                print("POS tag not in list:", pos, pos_tags)
                return None

        word_val_data = self.val_data[word][pos_tag]

        # add missing sentences + embeddings
        missing_sentences = [s for s in word_val_data['sentences'] if s not in word_data.sentences]
        if missing_sentences:
            word_data.add_missing_sentences(missing_sentences, [pos_tag] * len(missing_sentences), self.word_embeddings, word_val_data)

        word_data.val_ids = [word_data.sentences.index(s) for s in word_val_data['sentences']]
        word_data.validation_labels = word_val_data['labels']

        #if reduce_embeddings:
        #    word_data.embeddings = self.reduce_embeddings_umap(word_data.embeddings)

        return word_data

    """def reduce_embeddings_umap(self, embeddings: List[List[float]]) -> List[List[float]]:
        scaled_data = self.scaler.fit_transform(embeddings)
        scaled_data = self.reducer.fit_transform(scaled_data)
        return [[x if self.not_nan_or_inf(x) else 0.0 for x in s] for s in scaled_data]

    @staticmethod
    def not_nan_or_inf(a):
        # TODO: fix
        a = asarray(a, dtype=float)
        return a.dtype.char in 'efdgFDG' and not np.isfinite(a).all()"""

    @staticmethod
    def write_results(out_file: str, algorithm_params_list: List[algorithms.ClusteringAlgorithm]):
        """
        Write scores of given algorithms to 'out_file'.

        :param out_file: file to write scores to
        :param algorithm_scores: a dict of scores (dict of float per word) per algorithm
        :param algorithm_params_list:
        :return: None
        """

        best_algo = {'score': 0.0, 'algo': None}

        with open(out_file, "w", encoding="utf8") as f:
            for algo in algorithm_params_list:

                f.write("Score for %s\n" % algo.id)

                #for word in algo.score.keys():
                #    f.write("\t'%s': %s\n" % (word, str(algo.score[word])))

                avg_scores = get_avg_scores(algo.score, ['adjusted_rand','completeness', 'f1_score', 'silhouette'])
                f.write("Avg score: %s\n\n" % str(avg_scores))

                algo_score = avg_scores['adjusted_rand'][0]
                if algo_score and algo_score > best_algo['score']:

                    best_algo['score'] = algo_score
                    best_algo['algo'] = algo.id

            f.write("Best algorithm: %s score: %s" % (best_algo['algo'], best_algo['score']))

