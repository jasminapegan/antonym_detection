import networkx as nx
import partition_networkx
from networkx.algorithms import cluster
import community
from networkx.algorithms.link_analysis import pagerank
from scipy import spatial
import file_helpers


def build_graph(words, word_forms, sentences, embeddings, distance='cosine_similarity', n=1):

    node_data = [(i, {"word": words[i], "word_form": word_forms[i], "sentence": sentences[i], "embedding": embeddings[i]})
                 for i in range(len(words))]

    if distance == 'cosine_similarity':
        edge_data = [(i, j, spatial.distance.cosine(embeddings[i], embeddings[j]))
                     for i in range(len(words)) for j in range(len(words)) if i != j]

    elif distance == 'relative_cos_sim':
        edge_data = []

        for i in range(len(words)):

            cos_sim = [(j, spatial.distance.cosine(embeddings[i], embeddings[j])) for j in range(len(words)) if i != j]
            cos_sim.sort(key=lambda x: x[1])
            max_n_sum = sum([x[1] for x in cos_sim[-n:]])
            relative_cos_sim = [(j, d / max_n_sum) for j, d in cos_sim]

            edge_data += [(i, j, d) for j, d in relative_cos_sim]
    else:
        raise Exception("Unknown distance '%'." % distance)

    graph = nx.Graph()
    graph.add_nodes_from(node_data)
    graph.add_weighted_edges_from(edge_data)

    return graph


def pagerank_partitioning(graph, out_file):
    print(pagerank(graph))


def min_cut_partitioning(graph):
    graph = graph.to_directed()
    partition = nx.min_cost_flow(graph)

    #unwrapped_partition = {}
    #for key, dictionary in partition.keys():
    #    for k, v in dictionary:
    #        unwrapped_partition[k] = key

    return partition


def results_from_partition(graph, partition):
    sorted_partitions = list(partition.items())
    sorted_partitions.sort(key=lambda x: x[1])

    data = []
    nodes = graph.nodes(data=True)

    for node, label in sorted_partitions:
        node_data = nodes[node]
        data.append([label, node_data["word"], node_data["sentence"]])

    data.sort(key=lambda x: x[0])
    return data


def get_partitions_by_word(data_file, words_json_file, out_file, algorithm='community'):
    words_json = file_helpers.load_json_word_data(words_json_file)
    print("words json loaded")

    n_words = len(words_json.keys())
    idx = 0

    with open(out_file, "w", encoding="utf8") as outf:
        for i in range(len(words_json.keys())):

            words, word_forms, sentences, embeddings, idx = file_helpers.load_sentences_embeddings_file_grouped(data_file, start=idx)

            assert len(set(words)) == 1

            if words[0] == "iztočnica":
                continue

            word = words[0]
            print("parsed data: %s\t%d/%d" % (word, i, n_words))

            n_samples = len(sentences)
            n = len(words_json[word])

            if n == 1:
                results = list(zip([0] * n_samples, words, sentences))
                #file_helpers.write_grouped_data(outf, results)

            elif n_samples < n:
                print("Too little samples for %s: %d samples, %d partitions." % (word, n_samples, n))

            else:
                n_partitions = n #2 * (n+1)
                graph = build_graph(words, word_forms, sentences, embeddings, distance='relative_cos_sim', n=1)
                partition = None

                if algorithm == 'community':
                    partition = community.best_partition(graph)

                elif algorithm == 'min_cut':    # not working yet
                    partition = min_cut_partitioning(graph)

                #print(partition)
                results = results_from_partition(graph, partition)

                file_helpers.write_grouped_data(outf, results) #, centroids=clusterer.cluster_centers_)
