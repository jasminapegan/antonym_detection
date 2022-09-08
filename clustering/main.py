import file_helpers
from clustering import find_best, plotting, scoring, processing, algorithms
from clustering.find_sense import SenseClusters, print_missing_senses

#validation_embeddings = "../data/embeddings/val_embeddings_sorted.txt" # "../../embeddings/val_embeddings_sorted.txt"
from clustering.scoring import load_cluster_scores_dict, get_avg_scores

validation_embeddings = "../data/embeddings/val.txt" # "../../embeddings/val_embeddings_sorted.txt"
validation_embeddings2 = "../data/embeddings/lemmatized/val.txt" # "../../embeddings/val_embeddings_sorted.txt"
validation_data = "../data/dataset/bkp/val.txt" #"../data/dataset/val_sorted.txt" # "../../data/val.txt"
validation_words = "../data/dataset/bkp/val_words_all.txt" #"../data/dataset/val_words.txt" # "../../data/val_words.txt"
word_data ="../data/dataset/bkp/val_words_all.txt" #"../data/dataset/all_words.txt" #"../../data/all_words.txt"

test_embeddings = "../data/embeddings/archive/clustering_test.txt"
test_data = "../data/dataset/bkp/clustering_test.txt"
test_words = "../data/dataset/bkp/clustering_test_words.txt"

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
#find_best.find_best_all(validation_embeddings2, word_data, validation_data, output_vectors=True, use_pos=True)
find_best.find_best_all(test_embeddings, word_data, test_data, output_vectors=False, use_pos=False)

#find_best.ensemble_clustering(validation_embeddings, word_data, validation_data, output_vectors=True, out_dir='best')

result_files =  ["out_latest/agglomerative/agglomerative-affinity=precomputed,distance=relative_cosine,k=20_data.tsv",
                 "out_latest/kmeans/kmeans-algorithm=full,n_init=130_data.tsv",
                 "out_latest/spectral-affinity=cosine,n_neighbors=3_data.tsv"]

score_files =  ["out_latest/agglomerative/agglomerative-affinity=precomputed,distance=relative_cosine,k=20.tsv",
                "out_latest/kmeans/kmeans-algorithm=full,n_init=130.tsv",
                "out_latest/spectral/spectral-affinity=cosine,n_neighbors=3.tsv"]

#processing.find_best_clusters(score_files)

#plotting.plot_sentence_neighborhood(result_files, "mehkužec", None, "mehkuzec.png")
#plotting.plot_sentence_neighborhood(result_files, "orbita", None, "orbita.png")

#plotting.prepare_data(result_files, "mehkužec_data.tsv", "mehkužec_labels.tsv", words=["mehkužec"])
#plotting.prepare_data(result_files, "orbita_data.tsv", "orbita_labels.tsv", words=["orbita"])

#processing.compare_clusters(word_data, result_files, ["agglomerative", "kmeans", "spectral"],
#                            validation_data, validation_words, "out_latest/compare_clusters.tsv", "clusters_data_latest.tsv")

#evaluation.compare_clusters(["živo srebro", "oljčen", "administratorski", "kadriranje", "seviljska pomaranča", "ionosfera", "boksarski", "kros", "monsunski", "korintski", "mahagonijev", "karatejski", "plastično", "navadna borovnica", "arktičen", "mehiška", "pehtranov", "topolov", "manifest", "izbira", "pastorek", "širiti", "EMŠO", "amortizer", "angleški", "kutina", "etan", "hraniti", "tamarindov", "dvoletnica", "boks", "živčen", "atlet", "fokstrot", "abolicionist", "modrikavost", "tenis", "vziti", "brusiti", "vratarjenje", "artičoka", "ligenj", "tektonski", "divjačina", "oslič", "nadležno", "riž", "akumulativnost", "dohodek", "jahačica", "lešnikast", "gvajanski", "brin", "saški", "perutnička", "kumkvat", "maklura", "lizika", "vzhod", "peroksid", "zmikastiti", "eliptičen", "pasat", "kontraritem", "evrski", "analognost", "gozdna jagoda", "odbojka", "mašen", "abotnost", "trepalnica", "akacija", "žvečilec", "dren", "bordar", "bodibilderski", "ananas", "sednica", "banana", "rokomet", "žonglerka", "leskovina", "državljan", "gramofonski", "Tehtnica", "erg", "pokeraški", "Karavanke", "marelica", "brezov", "Ariadnina nit", "mandljevo mleko", "magnetofon", "fiziologija", "medicinka", "kočura", "pav", "slamnik"],
#                            result_files, ["agglomerative", "kmeans", "spectral"],
#                            validation_data, validation_words, "out/compare_clusters_chosen.tsv", "chosen_clusters.tsv")
#evaluation.compare_clusters(["brž", "izkoreninjen", "gsm", "šminka", "gnezdišče", "močvirje", "reprezentanca", "burleska", "noseč", "dramatično", "resolucija", "donatorka", "jahač", "ilustracija", "etničen", "bik", "geslo", "teflon", "farsa", "pastorka", "študent", "polovica", "deskar", "ikona", "osa", "središčnica", "aristokracija", "mehkužec"],
#                            result_files, ["agglomerative", "kmeans", "spectral"],
#                            validation_data, validation_words, "out/compare_clusters_good.tsv", "good_clusters.tsv")
#random_words = ["mehkužec", "električar", "žrtev", "mečarica", "grob", "bogokletstvo"]
#evaluation.compare_clusters(random_words, result_files, ["agglomerative", "kmeans", "spectral"],
#                            validation_data, validation_words, "out/compare_random_clusters.tsv", "random_clusters.tsv")

#processing.write_stats(score_files, result_files, "stats.tsv", "scores_plot_data_2.tsv")
#plotting.plot_data("scores_plot_data_2.tsv")

syn_ant_dataset = "../data/sources/syn_ant/syn_ant_dataset_all.tsv"
labeled_embeddings = "../data/embeddings/lemmatized/labeled_embeddings.txt"
sense_data = "../data/sources/sense/sense_data.txt"
f = "ant_syn_senses/new/"
#f2 = "ant_syn_senses/no_outliers/"

#SenseClusters(syn_ant_dataset, labeled_embeddings, sense_data, f + "ent_no_outl.txt", algo='weighed_entropy', clean_data=True)
#SenseClusters(syn_ant_dataset, labeled_embeddings, sense_data, f + "ent_with_outl.txt", algo='weighed_entropy', clean_data=False)
#SenseClusters(syn_ant_dataset, labeled_embeddings, sense_data, f"{f}avg_d_with_outl_k=10.txt", algo='avg_dist', clean_data=False, k=10)
#for ratio in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
#    SenseClusters(syn_ant_dataset, labeled_embeddings, sense_data, f"{f}avg_d_no_outl_ratio={ratio}.txt", algo='avg_dist', clean_data=True, ratio=ratio)

#SenseClusters(syn_ant_dataset, labeled_embeddings, sense_data, f"{f}avg_d_with_outl_uniform.txt", algo='avg_dist', weights='uniform')

#sc = SenseClusters(syn_ant_dataset, labeled_embeddings, sense_data, f"{f}min_avg_dist.txt", algo='min_avg_dist')
#sc.execute_algorithm(f"{f}min_dist.txt", algo='min_dist')
#sc.execute_algorithm(f"{f}avg_min_dist.txt", algo='avg_min_dist')
#sc.execute_algorithm(f"{f}avg_dist.txt", algo='avg_dist')

#sc = SenseClusters(syn_ant_dataset, labeled_embeddings, sense_data, f"{f}description_dist.txt", algo='description_dist', ignore_missing=True)

#scoring.evaluate_cluster_results("ant_syn_senses/archive/sopomenke_protipomenke/napovedi.txt", "ant_syn_senses/archive/sopomenke_protipomenke/ocene.txt", "ant_syn_senses/min_avg_dist.txt")
#scoring.evaluate_cluster_results("ant_syn_senses/archive/sopomenke_protipomenke/napovedi.txt", "ant_syn_senses/archive/sopomenke_protipomenke/ocene.txt", "ant_syn_senses/min_avg_dist.txt")
#scoring.evaluate_cluster_results("ant_syn_senses/archive/sopomenke_protipomenke/napovedi.txt", "ant_syn_senses/archive/sopomenke_protipomenke/ocene.txt", "ant_syn_senses/avg_min_dist.txt") # best: ok: 12, nok: 3
#scoring.evaluate_cluster_results("ant_syn_senses/archive/sopomenke_protipomenke/napovedi.txt", "ant_syn_senses/archive/sopomenke_protipomenke/ocene.txt", "ant_syn_senses/min_dist.txt")
#scoring.evaluate_cluster_results("ant_syn_senses/archive/sopomenke_protipomenke/napovedi.txt", "ant_syn_senses/archive/sopomenke_protipomenke/ocene.txt", "ant_syn_senses/avg_dist.txt")

#print_missing_senses(sense_data, labeled_embeddings, "ant_syn_senses/missing_examples.txt")


"""for f in score_files:
    print(f)
    score_dict = load_cluster_scores_dict(f)

    avg_scores = get_avg_scores(score_dict, ['adjusted_rand', 'completeness', 'f1_score', 'silhouette'])
    print("Avg score: %s\n\n" % str(avg_scores))"""