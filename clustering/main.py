from clustering import find_best

validation_embeddings = "data/embeddings.txt"
validation_data = "data/labeled.txt"
word_data = "data/words.txt"

# KMeans - params: n_clusters, n_init
# Spectral - params: n_clusters, separate: distance metric
# Hierarchical - params: n_clusters, linkage, affinity
#   complete linkage requires affinity param
# DBSCAN - params: eps, distance metric

#find_best.find_best_kmeans(validation_embeddings_sample, word_json_file, validation_data, "clusters/kmeans/")
find_best.find_best_kmeans(validation_embeddings, word_data, validation_data)
