import numpy as np
import umap
import matplotlib.pyplot as plt
from scipy import spatial
from sklearn.preprocessing import StandardScaler

from file_helpers import convert_to_np_array


def plot_sentence_neighborhood(data_file, word, sentence, n=None, note=""):
    print("reading data ...")
    data = read_word_data(data_file, word)
    print("processing data ...")
    embedding = data[sentence]['embedding']

    sentences = list(data.keys())
    embeddings = [data[s]['embedding'] for s in sentences]
    distances = [spatial.distance.cosine(e, embedding) for e in embeddings]

    if n is None:
        n = len(distances)
    else:
        n = min(n, len(distances))

    top_n = np.argpartition(distances, -n)[-n:]
    top_n_sentences = [sentences[i] for i in top_n]

    top_n_data = [{'sentence': s, **data[s]} for s in top_n_sentences]
    print("plotting ...")
    plot_cluster(top_n_data, "Sentence neighborhood: %s\n%s" % (sentence[:30], note))

def plot_cluster(data, title):
    sentences = [x['sentence'] for x in data]
    labels = list(set([x['label'] for x in data]))
    embeddings = [x['embedding'] for x in data]
    points = reduce_points(embeddings)

    cmap = plt.cm.get_cmap('hsv', len(labels))
    colors = [cmap(labels.index(x['label'])) for x in data]

    X = [p[0] for p in points]
    Y = [p[1] for p in points]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.scatter(X, Y, c=colors)

    for i, s in enumerate(sentences):
        ax.annotate(s[:25], (X[i], Y[i]))

    ax.legend()
    ax.set_title(title)

    plt.show()

def reduce_points(embeddings):
    reducer = umap.UMAP()
    scaled_data = StandardScaler().fit_transform(embeddings)
    data = reducer.fit_transform(scaled_data)
    return data


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

