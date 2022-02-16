import sklearn.cluster
from scipy import spatial

import file_helpers
import numpy as np
from sklearn import metrics


def parse_single_cluster_words(data_file, words_json_file, out_file, out_file_embeddings):
    words_json = file_helpers.load_json_word_data(words_json_file)
    idx = 0

    with open(out_file, "w", encoding="utf8") as outf:
        with open(out_file_embeddings, "w", encoding="utf8") as outf_embeddings:
            for i in range(len(words_json.keys())):

                word, n_samples, n, words, sentences, embeddings, idx = get_word_data(data_file, words_json, idx)

                if word == "iztočnica":
                    continue

                if n == 1:
                    results = list(zip([0] * n_samples, words, sentences))
                    file_helpers.write_grouped_data(outf, results)

                    centroid = [np.mean([e[i] for e in embeddings]) for i in range(len(embeddings[0]))]
                    results_embeddings = [[0, word, centroid]]
                    file_helpers.write_data_for_classification(outf_embeddings, results_embeddings)


def get_clusters_by_word(word, sentences, embeddings, n_samples, n, parameters):
    clusterer = parameters['clusterer']
    algorithm = parameters['algorithm']
    params = parameters['parameters']

    if n_samples < n:
        #print("Too little samples for %s: %d samples, %d clusters." % (word, n_samples, n))
        return None, None, clusterer

    else:
        n_clusters = min(n, n_samples)

        # KMeans - params: n_clusters, n_init
        if algorithm == 'kmeans':
            if clusterer is None:
                clusterer = sklearn.cluster.KMeans()

            clusterer.set_params(**params, n_clusters=n_clusters)
            labels = clusterer.fit_predict(embeddings)

        # Spectral - params: n_clusters, distance metric
        elif algorithm == 'spectral':
            if 'distance' in parameters.keys():
                distance = parameters['distance']
            else:
                distance = None

            if clusterer is None:
                clusterer = sklearn.cluster.SpectralClustering()
            clusterer.set_params(**params, n_clusters=n_clusters)

            if distance:

                if distance == 'relative_cosine':
                    k = parameters['k']
                    adj_matrix = relative_cosine_similarity(embeddings, k=k)
                elif distance == 'cosine':
                    adj_matrix = cosine_similarity(embeddings)
                else:
                    raise Exception("Unknown distance: %s" % distance)

                labels = clusterer.fit_predict(adj_matrix)

            else:
                labels = clusterer.fit_predict(embeddings)

        # Hierarchical - params: n_clusters, linkage, affinity
        elif algorithm == 'agglomerative':
            affinity = params['affinity']

            if clusterer is None:
                clusterer = sklearn.cluster.AgglomerativeClustering()

            clusterer.set_params(**params, n_clusters=n_clusters)

            if affinity == 'precomputed':
                k = parameters['k']
                adj_matrix = relative_cosine_similarity(embeddings, k)
                labels = clusterer.fit_predict(adj_matrix)
            else:
                labels = clusterer.fit_predict(embeddings)

        # DBSCAN - params: eps, distance metric
        elif algorithm == 'dbscan':

            if clusterer is None:
                clusterer = sklearn.cluster.DBSCAN()
            clusterer.set_params(**params)

            labels = clusterer.fit_predict(embeddings)

        else:
            raise Exception("Unknown algorithm: %s" % algorithm)

        out_file = parameters['out_file']
        if out_file:
            with open(out_file, "a", encoding="utf8") as outf:
                #out_data = list(zip(labels, [word] * n_samples, sentences)) #, embeddings))
                #file_helpers.write_grouped_data(outf, sorted(out_data, key=lambda x: x[0])) #, centroids=clusterer.cluster_centers_)

                str_embeddings = [" ".join([str(x) for x in e]) for e in embeddings]
                out_data = list(zip([str(x) for x in labels], [word] * n_samples, sentences, str_embeddings))
                outf.writelines(["\t".join(line) + "\n" for line in out_data])

        silhouette = None
        try:
            silhouette = metrics.silhouette_score(embeddings, labels, metric='cosine')
            #print("silhouette score: %f" % silhouette)
        except Exception as e:
            pass #print("silhouette score could not be calculated: %s" % str(e))

        return labels, silhouette, clusterer


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


def get_word_data(data_file, words_json, start_idx):
    words, sentences, embeddings, idx = file_helpers.load_sentences_embeddings_file_grouped(data_file, start=start_idx, only_word=True)

    assert len(set(words)) == 1

    word = words[0]
    n_samples = len(sentences)
    n = len(words_json[word])

    return word, n_samples, n, words, sentences, embeddings, idx