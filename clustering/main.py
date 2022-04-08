import file_helpers
from clustering import find_best, plotting, scoring, processing

validation_embeddings = "../data/embeddings/val_embeddings_sorted.txt" #"data/embeddings.txt" #
validation_data = "../data/dataset/val.txt" #"data/labeled.txt" #
validation_words = "../data/dataset/val_words.txt" #"data/labeled.txt" #
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
find_best.find_best_all(validation_embeddings, word_data, validation_data, output_vectors=True)

result_files =  ["out/agglomerative/agglomerative-affinity=precomputed,distance=relative_cosine,k=20_data.tsv",
                 "out/kmeans/kmeans-algorithm=full,n_init=130_data.tsv",
                 "out/spectral/spectral-affinity=cosine,n_neighbors=3_data.tsv"]

score_files =  ["out/agglomerative/agglomerative-affinity=precomputed,distance=relative_cosine,k=20.tsv",
                 "out/kmeans/kmeans-algorithm=full,n_init=130.tsv",
                 "out/spectral/spectral-affinity=cosine,n_neighbors=3.tsv"]

processing.find_best_clusters(score_files)

plotting.plot_sentence_neighborhood(result_files, "mehkužec", None, "mehkuzec.png")
plotting.plot_sentence_neighborhood(result_files, "orbita", None, "orbita.png")

#plotting.prepare_data(result_files, "mehkužec_data.tsv", "mehkužec_labels.tsv", words=["mehkužec"])
#plotting.prepare_data(result_files, "orbita_data.tsv", "orbita_labels.tsv", words=["orbita"])

#evaluation.compare_clusters(word_data, result_files, ["agglomerative", "kmeans", "spectral"],
#                            validation_data, validation_words, "out/compare_clusters.tsv", "clusters_data.tsv")

#evaluation.compare_clusters(["živo srebro", "oljčen", "administratorski", "kadriranje", "seviljska pomaranča", "ionosfera", "boksarski", "kros", "monsunski", "korintski", "mahagonijev", "karatejski", "plastično", "navadna borovnica", "arktičen", "mehiška", "pehtranov", "topolov", "manifest", "izbira", "pastorek", "širiti", "EMŠO", "amortizer", "angleški", "kutina", "etan", "hraniti", "tamarindov", "dvoletnica", "boks", "živčen", "atlet", "fokstrot", "abolicionist", "modrikavost", "tenis", "vziti", "brusiti", "vratarjenje", "artičoka", "ligenj", "tektonski", "divjačina", "oslič", "nadležno", "riž", "akumulativnost", "dohodek", "jahačica", "lešnikast", "gvajanski", "brin", "saški", "perutnička", "kumkvat", "maklura", "lizika", "vzhod", "peroksid", "zmikastiti", "eliptičen", "pasat", "kontraritem", "evrski", "analognost", "gozdna jagoda", "odbojka", "mašen", "abotnost", "trepalnica", "akacija", "žvečilec", "dren", "bordar", "bodibilderski", "ananas", "sednica", "banana", "rokomet", "žonglerka", "leskovina", "državljan", "gramofonski", "Tehtnica", "erg", "pokeraški", "Karavanke", "marelica", "brezov", "Ariadnina nit", "mandljevo mleko", "magnetofon", "fiziologija", "medicinka", "kočura", "pav", "slamnik"],
#                            result_files, ["agglomerative", "kmeans", "spectral"],
#                            validation_data, validation_words, "out/compare_clusters_chosen.tsv", "chosen_clusters.tsv")
#evaluation.compare_clusters(["brž", "izkoreninjen", "gsm", "šminka", "gnezdišče", "močvirje", "reprezentanca", "burleska", "noseč", "dramatično", "resolucija", "donatorka", "jahač", "ilustracija", "etničen", "bik", "geslo", "teflon", "farsa", "pastorka", "študent", "polovica", "deskar", "ikona", "osa", "središčnica", "aristokracija", "mehkužec"],
#                            result_files, ["agglomerative", "kmeans", "spectral"],
#                            validation_data, validation_words, "out/compare_clusters_good.tsv", "good_clusters.tsv")
#random_words = ["mehkužec", "električar", "žrtev", "mečarica", "grob", "bogokletstvo"]
#evaluation.compare_clusters(random_words, result_files, ["agglomerative", "kmeans", "spectral"],
#                            validation_data, validation_words, "out/compare_random_clusters.tsv", "random_clusters.tsv")

processing.write_stats(score_files, "scores.tsv")