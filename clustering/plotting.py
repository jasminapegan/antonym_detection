import os
import numpy as np
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from adjustText import adjust_text
from scipy import spatial
from sklearn.preprocessing import StandardScaler

from clustering.processing import get_centroid
from file_helpers import convert_to_np_array


def plot_sentence_neighborhood(data_files, word, sentence, filename, n=None, cutoff_sentence=50, figsize=(10, 10)):
    for data_file in data_files:
        directory = os.path.dirname(data_file)
        filepath = os.path.join(directory, filename)

        data = read_word_data(data_file, word)
        sentences = list(data.keys())
        embeddings = [{'sentence': s, **data[s]} for s in sentences]

        if not sentence:
            plot_cluster(embeddings, "All sentences, clustering: %s" % data_file,
                         filepath, cutoff_sentence=cutoff_sentence, figsize=figsize)

        else:
            embedding = data[sentence]['embedding']
            distances = [spatial.distance.cosine(e['embedding'], embedding) for e in embeddings]

            n = len(distances) if not n else min(n, len(distances))
            top_n = np.argpartition(distances, n)[:n]
            top_n_sentences = [sentences[i] for i in top_n]
            top_n_data = [{'sentence': s, **data[s]} for s in top_n_sentences]

            plot_cluster(top_n_data, "%d neighbors for: %s\n%s" % (n, sentence, data_file),
                         filepath, cutoff_sentence=cutoff_sentence, figsize=figsize)

def plot_cluster(data, title, filename, cutoff_sentence=50, figsize=(10, 10)):
    labels = list(set([x['label'] for x in data]))
    cmap = plt.cm.get_cmap('hsv', len(labels)+1)
    centroids = get_centroids(data)
    keys = list(centroids.keys())

    # reduce data with CMAP
    embeddings = [x['embedding'] for x in data]
    points, centroids_points = reduce_points(embeddings, [centroids[k] for k in keys])
    centroids = {k: centroids_points[i] for i, k in enumerate(keys)}
    for x, p in zip(data, points):
        x['point'] = p

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    texts = []

    for label in labels:
        color = cmap(labels.index(label))
        sentences = [x['sentence'] for x in data if x['label'] == label]
        points = [x['point'] for x in data if x['label'] == label]

        X = [p[0] for p in points]
        Y = [p[1] for p in points]

        # plot embeddings data
        ax.scatter(X, Y, c=[color for _ in sentences])
        texts += [ax.text(X[i], Y[i], s[:min(len(s), cutoff_sentence)]) for i, s in enumerate(sentences)]

        # plot centroid
        cx, cy = centroids[label]
        ax.scatter([cx], [cy], c=[color], marker="*")
        texts.append(ax.text(cx, cy, "centroid"))

    print("Adjusting text...")
    adjust_text(texts, lim=50, precision=0.15, force_text=1.0, arrowprops=dict(arrowstyle='->', color='grey'))
    print("Text adjusted")

    ax.legend(handles=[mpatch.Patch(color=cmap(i), label=labels[i]) for i in range(len(labels))])
    ax.set_title(title)

    plt.savefig(filename)
    print("Saved to file %s" % filename)

def get_centroids(data):
    X = [d['embedding'] for d in data]
    y = [d['label'] for d in data]

    clusters = {c: [] for c in y}
    for c, x in zip(y, X):
        clusters[c].append(x)

    centroids = {c: get_centroid(x) for c, x in clusters.items()}
    return centroids

def reduce_points(embeddings, centroids):
    reducer = umap.UMAP()
    scaled_data = StandardScaler().fit_transform(embeddings + centroids)
    data = reducer.fit_transform(scaled_data)
    return data[:-len(centroids)], data[-len(centroids):]

def read_word_data(data_file, word):
    with open(data_file, "r", encoding="utf8") as f:
        data = {}

        for line in f:
            label, w, sentence, embedding = line.strip().split("\t")

            if w == word:
                data[sentence] = {'label': int(label),
                                 'embedding': convert_to_np_array([embedding])[0]}
            elif w != word and data != {}:
                return data

    return data

def plot_data(data_file):
    with open(data_file, "r", encoding="utf8") as f:
        data = f.readlines()

    for line in data:
        if "Algorithm:" in line:
            algo = line.split(":")[1].strip()


        elif "Score names:" in line:
            score_names = line.split(":")[1].strip().split(" ")
            n = len(score_names)
            cmap = plt.cm.get_cmap('hsv', n + 1)

        elif "Score:" in line:
            score_name = line.split(":")[1].strip()

            if score_name not in score_names:
                score_names.append(score_name)

            # begin new plot
            plt.title = score_name
            plt.figure(figsize=(20, 10))
            plt.subplot(121)
            plt.xticks([0.1 * i for i in range(10)])
            i = score_names.index(score_name)
            c = cmap(i)

        else:
            stats_data, score_data = line.strip().split("\t")

            stats_data_split = stats_data.split(" ")
            score_data_split = score_data.split(" ")

            for x, y in zip(stats_data_split, score_data_split):
                if x == 'None' or y == 'None':
                    stats_data_split.remove(x)
                    score_data_split.remove(y)

            X = [float(x) for x in stats_data_split]
            Y = [float(x) for x in score_data_split]

            plt.scatter(X, Y, c=[c for _ in X], s=20)
            plt.title = algo
            plt.legend(bbox_to_anchor=(0, 0), loc='center right',
                       handles=[mpatch.Patch(color=cmap(i), label=score_names[i]) for i in range(len(score_names))])

result_files =  ["out_latest/agglomerative/agglomerative-affinity=precomputed,distance=relative_cosine,k=20_data.tsv",
                 "out_latest/kmeans/kmeans-algorithm=full,n_init=130_data.tsv",
                 "out_latest/spectral/spectral-affinity=cosine,n_neighbors=3_data.tsv"]

if __name__ == '__main__':
    plot_sentence_neighborhood(result_files, "električar", None, "elektricar.png", cutoff_sentence=60, figsize=(20,20))

    #plot_sentence_neighborhood(result_files, "mehkužec", None, "mehkuzec.png", cutoff_sentence=60, figsize=(20,20))
    #plot_sentence_neighborhood(result_files, "orbita", None, "orbita.png", cutoff_sentence=60, figsize=(20,20))
