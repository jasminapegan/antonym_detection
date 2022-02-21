""" Version 2 treats number of clusters as an unknown parameter. """
import file_helpers
from typing import List
from clustering import algorithms
from sklearn import metrics

from data import word


def kmeans_get_clusters_by_word(word, sentences, embeddings, n_samples, n, params, out_file=None, clusterer=None):
    n_clusters = min(n, n_samples)

    clusterer.set_params(**params, n_clusters=n_clusters)
    labels = clusterer.fit_predict(embeddings)

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
        # print("silhouette score: %f" % silhouette)
    except Exception as e:
        print("silhouette score could not be calculated: %s" % str(e))

    return labels, silhouette


def get_clusters_by_word_v2(word_data: word.WordData,
                            n: int,
                            algorithm: algorithms.ClusteringAlgorithm,
                            out_dir: str) -> (List[int], float):

    n_samples = word_data.n_sentences
    if n_samples < n:
        #print("Too little samples for %s: %d samples, %d clusters." % (word_data.word, n_samples, n))
        return None, None

    else:
        n_clusters = min(n, n_samples)

        labels = algorithm.predict(word_data.embeddings, n_clusters)

        out_file = out_dir + algorithm.get_algorithm_id()
        with open(out_file, "a", encoding="utf8") as outf:
            #out_data = list(zip(labels, [word] * n_samples, sentences)) #, embeddings))
            #file_helpers.write_grouped_data(outf, sorted(out_data, key=lambda x: x[0])) #, centroids=clusterer.cluster_centers_)

            str_embeddings = [" ".join([str(x) for x in e]) for e in word_data.embeddings]
            out_data = list(zip([str(x) for x in labels], [word_data.word] * n_samples, word_data.sentences, str_embeddings))
            outf.writelines(["\t".join(line) + "\n" for line in out_data])

        silhouette = None
        try:
            silhouette = metrics.silhouette_score(word_data.embeddings, labels, metric='cosine')
            #print("silhouette score: %f" % silhouette)
        except Exception as e:
            print("silhouette score could not be calculated: %s" % str(e))

        return labels, silhouette


def get_word_data(data_file, words_json, start_idx):
    words, sentences, embeddings, idx = file_helpers.load_sentences_embeddings_file_grouped(data_file, start=start_idx, only_word=True)

    assert len(set(words)) == 1

    word = words[0]
    n_samples = len(sentences)
    n = len(words_json[word])

    return word, n_samples, n, words, sentences, embeddings, idx