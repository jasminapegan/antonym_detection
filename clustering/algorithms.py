import os
from itertools import product
from typing import Dict, List

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from scipy import spatial

from data.word import WordData
from clustering.scoring import score_clustering


class ClusteringAlgorithm:

    def __init__(self, name: str, parameters: Dict, out_dir: str='out'):
        self.name = name
        self.parameters = parameters
        self.clusterer = None
        self.n_clusters = -1
        self.score = {}
        self.id = self.__get_algorithm_id__()

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        # create algorithm specific directory
        out_dir_algo = os.path.join(out_dir, name)
        if not os.path.exists(out_dir_algo):
            os.mkdir(out_dir_algo)

        # assure that file will be empty
        out_file = os.path.join(out_dir_algo, self.id + ".csv")
        if os.path.exists(out_file):
            os.remove(out_file)

        self.out_file = out_file

    def __get_algorithm_id__(self):
        """ Get unique string representing algorithm with all the parameters. """
        params = self.parameters.copy()
        if 'n_clusters' in params.keys():
            del params['n_clusters']
        if 'clusterer' in params.keys():
            del params['clusterer']
        parameters = list(params.items())
        parameters.sort()
        return  self.name + "-" + ",".join([str(k) + "=" + str(v) for k, v in parameters])

    def predict(self, embeddings: List[List[float]], n_clusters: int):
        pass

    def score_method(self, word_data: WordData, silhouette: float, n_clusters: int):
        """
        Scores method on 'word_data' and updates score parameter.

        :param word_data: chosen word data
        :param silhouette: silhouette score
        :param n_clusters: number of clusters
        :return: None
        """

        n_samples = word_data.n_sentences
        val_labels = word_data.validation_labels
        pred_labels = word_data.predicted_labels

        #adj_rand_score, completeness_score, f1_score, labels, confusion_matrix = \
        adj_rand_score, completeness_score = score_clustering(pred_labels, val_labels)

        #score_data = [silhouette, adj_rand_score, completeness_score, f1_score, n_samples, len(pred_labels)]
        score_data = [word_data.word, silhouette, adj_rand_score, completeness_score, n_samples, len(pred_labels)]

        score_dict = {
            'silhouette': silhouette,
            'adjusted_rand': adj_rand_score,
            'completeness': completeness_score
        }

        with open(self.out_file, "a", encoding="utf8") as f:
            f.write(", ".join([str(x) for x in score_data]) + "\n")
        self.score[word_data.word] = score_dict
        """
        if word_data.word not in self.score.keys():
            self.score[word_data.word] = {n_clusters: score_dict}
        else:
            self.score[word_data.word][n_clusters] = score_dict"""

    def get_best_n_clusters(self, word_data):
        scores = self.score[word_data.word].items()
        sorted_scores = list(scores).sort(key=lambda x: x[1])
        print(sorted_scores)
        print("Best n clusters: %d clusters %f silhouette score" *sorted_scores[-1])

class KMeansAlgorithm(ClusteringAlgorithm):

    def __init__(self, parameters: Dict):

        ClusteringAlgorithm.__init__(self, 'kmeans', parameters)

        if 'random_state' not in parameters.keys():
            parameters['random_state'] = 42
        if 'n_init' not in parameters.keys():
            parameters['n_init'] = 20
        if 'algorithm' not in parameters.keys():
            parameters['algorithm'] = 'elkan'

        self.clusterer = KMeans(**self.parameters)


    @staticmethod
    def get_clusterer_list(algorithms: List[str]=['elkan'], n_inits: List[int]=[20]):
        """ Get a list of clusterers with all possible combinations of given argument ranges. """
        return [
            KMeansAlgorithm({'n_init': n, 'algorithm': a}) for n, a in product(n_inits, algorithms)
        ]

    def predict(self, embeddings: List[List[float]], n_clusters: int):
        params = self.parameters
        params['n_clusters'] = n_clusters
        self.clusterer.set_params(**params)
        return self.clusterer.fit_predict(embeddings)


class SpectralAlgorithm(ClusteringAlgorithm):

    def __init__(self, parameters: Dict):
        affinity = parameters['affinity']
        assert affinity in ['cosine', 'nearest_neighbors', 'precomputed', 'rbf']

        ClusteringAlgorithm.__init__(self, 'spectral', parameters)

        if 'random_state' not in parameters.keys():
            parameters['random_state'] = 42
        if 'n_neighbors' not in parameters.keys():
            parameters['n_neighbors'] = 5

        if affinity == 'precomputed':
            assert parameters['distance'] == 'relative_cosine'
            if 'k' not in parameters.keys():
                parameters['k'] = 3

        elif affinity == 'rbf':
            if 'gamma' not in parameters.keys():
                parameters['gamma'] = 1.0

        clustering_params = {k: v for k, v in parameters.items() if k not in ['distance', 'k']}
        self.clusterer = SpectralClustering(**clustering_params)


    @staticmethod
    def get_clusterer_list(affinity: List[str]=['cosine'], n_neighbors: List[int]=[5], distance: List[str]=['cosine'],
                           ks: List[int]=[3], gamma: List[float]=[1.0]):
        """ Get a list of clusterers with all possible combinations of given argument ranges. """

        algorithms = []

        if 'precomputed' in affinity: # and 'relative_cosine' in distance:
            algorithms += [
                SpectralAlgorithm({
                    'affinity': 'precomputed',
                    'distance': 'relative_cosine',
                    'k': k,
                    'n_neighbors': n
                }) for k, n in product(ks, n_neighbors)]
            affinity.remove('precomputed')

        if 'rbf' in affinity:
            algorithms += [
                SpectralAlgorithm({
                    'affinity': 'rbf',
                    'gamma': g,
                    'n_neighbors': n
                }) for g, n in product(gamma, n_neighbors)
            ]
            affinity.remove('rbf')

        algorithms += [
            SpectralAlgorithm({
                'affinity': a,
                'n_neighbors': n
            }) for a, n in product(affinity, n_neighbors)
        ]

        return algorithms

    def predict(self, embeddings: List[List[float]], n_clusters: int):
        if 'distance' in self.parameters.keys():
            distance = self.parameters['distance']
        else:
            distance = None

        clustering_params = {k: v for k, v in self.parameters.items() if k not in ['distance', 'k']}
        clustering_params['n_clusters'] = n_clusters
        self.clusterer.set_params(**clustering_params)

        if distance:

            if distance == 'relative_cosine':
                k = self.parameters['k']
                adj_matrix = relative_cosine_similarity(embeddings, k=k)
            elif distance == 'cosine':
                adj_matrix = cosine_similarity(embeddings)
            else:
                raise Exception("Unknown distance: %s" % distance)

            return self.clusterer.fit_predict(adj_matrix)


        return self.clusterer.fit_predict(embeddings)


class AgglomerativeAlgorithm(ClusteringAlgorithm):

    def __init__(self, parameters: Dict):
        affinity = parameters['affinity']
        assert affinity in ['cosine', 'euclidean', 'l1', 'precomputed']

        ClusteringAlgorithm.__init__(self, 'agglomerative', parameters)

        #if 'random_state' not in parameters.keys():
        #    parameters['random_state'] = 42

        if affinity == 'precomputed':
            if 'k' not in parameters.keys():
                parameters['k'] = 3

        else:
            linkage = parameters['linkage']
            assert linkage in ['complete', 'average', 'ward']

            if linkage == 'ward':
                assert affinity == 'euclidean'

        clusterer_params = {k: v for k, v in self.parameters.items() if k not in ['k', 'distance']}
        self.clusterer = AgglomerativeClustering(**clusterer_params)

    def set_n_clusters(self, n: int):
        self.clusterer.n_clusters = n

    @staticmethod
    def get_clusterer_list(affinity: List[str]=['cosine'], linkage: List[str]=['complete'], distance: List[str]=[], ks: List[int]=[3]):
        """ Get a list of clusterers with all possible combinations of given argument ranges. """

        algorithms = []

        if 'precomputed' in affinity and 'relative_cosine' in distance:
            algorithms += [
                AgglomerativeAlgorithm({
                    'affinity': 'precomputed',
                    'distance': 'relative_cosine',
                    'k': k,
                }) for k in ks]

        if 'ward' in linkage and 'euclidean' in affinity:
            algorithms += [
                AgglomerativeAlgorithm({
                    'affinity': 'euclidean',
                    'linkage': 'ward'
                })
            ]

        algorithms += [
            AgglomerativeAlgorithm({
                'affinity': a,
                'linkage': l
            }) for a, l in product([a for a in affinity if a != 'precomputed'], [l for l in linkage if l != 'ward'])
        ]

        return algorithms

    def predict(self, embeddings: List[List[int]], n_clusters: int):
        affinity = self.parameters['affinity']

        clusterer_params = {k: v for k, v in self.parameters.items() if k not in ['k', 'distance']}
        clusterer_params['n_clusters'] = n_clusters
        if self.parameters['affinity'] == 'precomputed':
            clusterer_params['linkage'] = 'average'
        self.clusterer.set_params(**clusterer_params)

        if affinity == 'precomputed':
            if self.parameters['distance'] == 'relative_cosine':
                k = self.parameters['k']
                adj_matrix = relative_cosine_similarity(embeddings, k)
                return self.clusterer.fit_predict(adj_matrix)
            else:
                raise Exception("Unknown precomputed distance %s" % self.parameters['distance'])
        else:
            return self.clusterer.fit_predict(embeddings)


class DbscanAlgorithm(ClusteringAlgorithm):

    def __init__(self, parameters: Dict):

        ClusteringAlgorithm.__init__(self, 'dbscan', parameters)

        if 'eps' not in parameters.keys():
            parameters['eps'] = 0.5
        if 'metric' not in parameters:
            parameters['metric'] = 'cosine'
        if 'min_samples' not in parameters:
            parameters['min_samples'] = 5
        if 'leaf_size' not in parameters:
            parameters['leaf_size'] = 3

        self.clusterer = DBSCAN(**self.parameters)

    @staticmethod
    def get_clusterer_list(eps: List[float]=[0.5], min_samples: List[int]=[5], leaf_size:List[int]=[3]):
        """ Get a list of clusterers with all possible combinations of given argument ranges. """

        return [DbscanAlgorithm({
            'eps': e,
            'min_samples': s,
            'leaf_size': l
        }) for e, s, l in product(eps, min_samples, leaf_size)]

    def predict(self, embeddings: List[List[int]], n_clusters: int):
        self.clusterer.set_params(**self.parameters)
        return self.clusterer.fit_predict(embeddings)


def relative_cosine_similarity(data, k=1):
    n = len(data)
    k = min(k, n)
    cos_sim = [[spatial.distance.cosine(data[i], data[j]) for i in range(n)] for j in range(n)]
    cos_sim_sorted = [sorted(line) for line in cos_sim]
    max_k_sum = [sum(line[-k:]) for line in cos_sim_sorted]
    relative_cos_sim = [d / s for d, s in zip(cos_sim, max_k_sum)]
    for i in range(n):
        relative_cos_sim[i][i] = 1
    return relative_cos_sim


def cosine_similarity(data):
    n = len(data)
    cos_sim = [[spatial.distance.cosine(data[i], data[j]) for i in range(n)] for j in range(n)]
    for i in range(n):
        cos_sim[i][i] = 1
    return cos_sim
