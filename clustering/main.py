import file_helpers
from clustering import find_best, plotting, scoring, evaluation

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

result_files =  ["out/agglomerative/agglomerative-affinity=precomputed,distance=relative_cosine,k=20_data.tsv",
                 "out/kmeans/kmeans-algorithm=full,n_init=130_data.tsv",
                 "out/spectral/spectral-affinity=cosine,n_neighbors=3_data.tsv"]

score_files =  ["out/agglomerative/agglomerative-affinity=precomputed,distance=relative_cosine,k=20.tsv",
                 "out/kmeans/kmeans-algorithm=full,n_init=130.tsv",
                 "out/spectral/spectral-affinity=cosine,n_neighbors=3.tsv"]

#evaluation.find_best_clusters(score_files)
#Correct - common: 5 words: nadležno, had, angleški, kahla, bobnič
#Close - common: 5 words: nadležno, had, angleški, kahla, bobnič
#words: 1132, 1121, 1119

#plotting.plot_sentence_neighborhood(result_files, "Amazonka", None, "Amazonka.png")
#plotting.prepare_data(result_files, "amazonka_data.tsv", "amazonka_labels.tsv", words=["Amazonka"])
