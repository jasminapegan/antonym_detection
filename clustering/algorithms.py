from itertools import product
from typing import Dict, List

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from scipy import spatial

from data.word import WordData
from clustering.scoring import score_clustering


class ClusteringAlgorithm:

    def __init__(self, name: str, parameters: Dict):
        self.name = name
        self.parameters = parameters
        self.clusterer = None
        self.n_clusters = -1
        self.score = {}

    def get_algorithm_id(self):
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

    def score_method(self, word_data: WordData, silhouette: float):
        """
        Scores method on 'word_data' and updates score parameter.

        :param word_data: chosen word data
        :param silhouette: silhouette score
        :return: None
        """

        n_samples = word_data.n_sentences
        val_labels = word_data.validation_labels
        pred_labels = word_data.predicted_labels

        rand_score, adj_rand_score, completeness_score, f1_score, labels, confusion_matrix = \
            score_clustering(pred_labels, val_labels)

        #print("f1 score: %f" % f1_score)
        #print(confusion_matrix)

        algo_id = self.get_algorithm_id()
        if algo_id not in self.score.keys():
            self.score[algo_id] = {}

        self.score[algo_id][word_data.word] = {
            'silhouette': silhouette,
            'rand': rand_score,
            'adjusted_rand': adj_rand_score,
            'completeness': completeness_score,
            'f1_score': f1_score,
            'n_samples': n_samples,
            'n_non_null': len(pred_labels)
        }


class KMeansAlgorithm(ClusteringAlgorithm):

    def __init__(self, parameters: Dict):
        self.name = 'kmeans'
        self.parameters = parameters
        self.score = {}

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

        self.name = 'spectral'
        self.parameters = parameters
        self.score = {}

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

        self.clusterer = SpectralClustering(**self.parameters)


    @staticmethod
    def get_clusterer_list(affinity: List[str]=['cosine'], n_neighbors: List[int]=[5], distance: List[str]=['cosine'],
                           ks: List[int]=[3], gamma: List[float]=[1.0]):
        """ Get a list of clusterers with all possible combinations of given argument ranges. """

        algorithms = []

        if 'precomputed' in affinity and 'relative_cosine' in distance:
            algorithms += [
                SpectralAlgorithm({
                    'affinity': 'precomputed',
                    'distance': 'relative_cosine',
                    'k': k,
                    'n_neighbors': n
                }) for k, n in product(ks, n_neighbors)]

        if 'rbf' in affinity:
            algorithms += [
                SpectralAlgorithm({
                    'gamma': g,
                    'n_neighbors': n
                }) for g, n in product(gamma, n_neighbors)
            ]

        algorithms += [
            SpectralAlgorithm({
                'affinity': affinity,
                'n_neighbors': n
            }) for a, n in product(affinity, n_neighbors)
        ]

        return algorithms

    def predict(self, embeddings: List[List[float]], n_clusters: int):
        if 'distance' in self.parameters.keys():
            distance = self.parameters['distance']
        else:
            distance = None

        params = self.parameters
        params['n_clusters'] = n_clusters
        self.clusterer.set_params(**params)

        if distance:

            if distance == 'relative_cosine':
                k = self.parameters['k']
                adj_matrix = relative_cosine_similarity(embeddings, k=k)
            elif distance == 'cosine':
                adj_matrix = cosine_similarity(embeddings)
            else:
                raise Exception("Unknown distance: %s" % distance)

            return self.clusterer.fit_predict(adj_matrix, n_clusters=n_clusters)

        return self.clusterer.fit_predict(embeddings, n_clusters=n_clusters)


class AgglomerativeAlgorithm(ClusteringAlgorithm):

    def __init__(self, parameters: Dict):
        affinity = parameters['affinity']
        assert affinity in ['cosine', 'euclidean', 'nearest_neighbors']

        self.name = 'agglomerative'
        self.parameters = parameters
        self.score = {}

        if 'random_state' not in parameters.keys():
            parameters['random_state'] = 42

        if affinity == 'precomputed':
            if 'k' not in parameters.keys():
                parameters['k'] = 3

        else:
            linkage = parameters['linkage']
            assert linkage in ['complete', 'average', 'ward']

            if linkage == 'ward':
                assert affinity == 'euclidean'

        self.clusterer = AgglomerativeClustering(**self.parameters)

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
                    'linkage': 'relative_cosine',
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
            }) for a, l in product(affinity, [l for l in linkage if l != 'ward'])
        ]

        return algorithms

    def predict(self, embeddings: List[List[int]], n_clusters: int):
        affinity = self.parameters['affinity']

        params = self.parameters
        params['n_clusters'] = n_clusters
        self.clusterer.set_params(**params)

        if affinity == 'precomputed':
            k = self.parameters['k']
            adj_matrix = relative_cosine_similarity(embeddings, k)
            return self.clusterer.fit_predict(adj_matrix)
        else:
            return self.clusterer.fit_predict(embeddings)


class DbscanAlgorithm(ClusteringAlgorithm):

    def __init__(self, parameters: Dict):

        self.name = 'agglomerative'
        self.parameters = parameters
        self.score = {}

        if 'random_state' not in parameters.keys():
            parameters['random_state'] = 42
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
