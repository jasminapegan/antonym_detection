import file_helpers
from clustering import find_best, plotting, scoring

validation_embeddings = "../data/embeddings/val_embeddings_sorted.txt" #"data/embeddings.txt" #
validation_data = "../data/dataset/val.txt" #"data/labeled.txt" #
word_data = "../data/dataset/all_words.txt" #"data/words.txt" #

# KMeans - params: n_clusters, n_init
# Spectral - params: n_clusters, separate: distance metric
# Hierarchical - params: n_clusters, linkage, affinity
#   complete linkage requires affinity param
# DBSCAN - params: eps, distance metric

#find_best.find_best_kmeans(validation_embeddings_sample, word_json_file, validation_data, "clusters/kmeans/")


#find_best.find_best_kmeans(validation_embeddings, word_data, validation_data)
#find_best.find_best_spectral(validation_embeddings, word_data, validation_data)
#find_best.find_best_agglomerative(validation_embeddings, word_data, validation_data)
#find_best.find_best_dbscan(validation_embeddings, word_data, validation_data, output_vectors=True)
#find_best.process_results("processed_data.xlsx", "out/kmeans")
#find_best.find_best_all(validation_embeddings, word_data, validation_data)
"""
find_best.get_results(["results/agglomerative"], "results/agglomerative")
find_best.get_results(["results/kmeans"], "results/kmeans")
find_best.get_results(["results/spectral"], "results/spectral")

with open("results/agglomerative/data.tsv", "r", encoding="utf8") as f:
    x = f.readlines()[2]
    y = x.split(" - ")[1]
    words1 = y.split(",")
with open("results/kmeans/data.tsv", "r", encoding="utf8") as f:
    x = f.readlines()[2]
    y = x.split(" - ")[1]
    words2 = y.split(",")
with open("results/spectral/data.tsv", "r", encoding="utf8") as f:
    x = f.readlines()[2]
    y = x.split(" - ")[1]
    words3 = y.split(",")

common = [x for x in words1 if x in words2 and x in words3]
print("skupne:", len(common))
print(common)
print(len(words1))
print(len(words2))
print(len(words3))
"""
"""
plotting.plot_sentence_neighborhood(
    "out/agglomerative/agglomerative-affinity=precomputed,distance=relative_cosine,k=20_data.tsv",
    "Amazonka",
    "\" V desetih letih je Earth Hour pomagal zavarovati morja v Rusiji in Argentini , zbiral sredstva za projekte ohranjanja narave v jugovzhodni Aziji in Amazonki in ustvaril povsem nov gozd v Ugandi .",
    n=35,
    note="agglomerative"
)
plotting.plot_sentence_neighborhood(
    "out/kmeans/kmeans-algorithm=full,n_init=130_data.tsv",
    "Amazonka",
    "\" V desetih letih je Earth Hour pomagal zavarovati morja v Rusiji in Argentini , zbiral sredstva za projekte ohranjanja narave v jugovzhodni Aziji in Amazonki in ustvaril povsem nov gozd v Ugandi .",
    n=35,
    note="kmeans"
)
plotting.plot_sentence_neighborhood(
    "out/spectral/spectral-affinity=cosine,n_neighbors=3_data.tsv",
    "Amazonka",
    "\" V desetih letih je Earth Hour pomagal zavarovati morja v Rusiji in Argentini , zbiral sredstva za projekte ohranjanja narave v jugovzhodni Aziji in Amazonki in ustvaril povsem nov gozd v Ugandi .",
    n=35,
    note="spectral"
)

file_helpers.filter_file_by_words(
    "out/agglomerative/agglomerative-affinity=precomputed,distance=relative_cosine,k=20_data.tsv",
    ["Amazonka"],
    "out/agglomerative/clusters.tsv",
    word_idx=1,
    skip_idx=3
)
file_helpers.filter_file_by_words(
    "out/kmeans/kmeans-algorithm=full,n_init=130_data.tsv",
    ["Amazonka"],
    "out/kmeans/clusters.tsv",
    word_idx=1,
    skip_idx=3
)
file_helpers.filter_file_by_words(
    "out/spectral/spectral-affinity=cosine,n_neighbors=3_data.tsv",
    ["Amazonka"],
    "out/spectral/clusters.tsv",
    word_idx=1,
    skip_idx=3
)
"""
scoring.compare_clusters(
    word_data,
    #["Amazonka", "kronika", "peroralen", "pevka"],
    ["out/agglomerative/agglomerative-affinity=precomputed,distance=relative_cosine,k=20_data.tsv",
     "out/kmeans/kmeans-algorithm=full,n_init=130_data.tsv",
     "out/spectral/spectral-affinity=cosine,n_neighbors=3_data.tsv"],
    ["agglomerative", "kmeans", "spectral"],
    validation_data,
    "out/compare_clusters.tsv"
)

