from clustering import clusters, find_best, clusters_v2, find_best_v2

#word_embeddings = "../data/embeddings/sample_sentences_100_sorted.txt"
#word_embeddings_v2 = "../data/embeddings/slohun/validation_embeddings.txt"
from data import grouped_file

validation_embeddings = "data/validation_dataset_complete_sorted.txt"
validation_embeddings_sample = "data/validation_sample.txt"
validation_data = "../data/sources/slohun/validation_sentences.txt"
word_json_file = "../data/sources/vse_besede.json"


"""
kmeans.get_kmeans_clusters(word_embeddings, "kmeans/test.txt")

graph = graph_partitions.build_graph(word_embeddings)
graph_partitions.pagerank_partitioning(graph, "pagerank/test.txt")
graph_partitions.community_partitioning(graph, "community/test.txt")
ensemble_clustering.mdec_ensemble_clustering(word_embeddings, "test.txt", 5.1)
"""

#clusters.get_clusters_by_word(word_embeddings, word_json_file, "clusters/v2/testSpectral.txt", algorithm='spectral')
#clusters.get_clusters_by_word(word_embeddings, word_json_file, "clusters/v2/testAgglomerativeWard.txt", algorithm='agglomerative', linkage='ward')
#clusters.get_clusters_by_word(word_embeddings, word_json_file, "clusters/v2/testAgglomerativeWard.txt", algorithm='agglomerative', linkage='complete')
#clusters.get_clusters_by_word(word_embeddings, word_json_file, "clusters/v2/testKnn.txt", algorithm='kmeans')
#clusters.get_clusters_by_word(word_embeddings, word_json_file, "clusters/v2/testKnn_more.txt", algorithm='kmeans')
#clusters.get_clusters_by_word(word_embeddings, word_json_file, "clusters/v2/testDbscan.txt", algorithm='dbscan')

"""
clusters.get_clusters_by_word("../data/embeddings/v2/sample_sentences_neposredno.txt", word_json_file, validation_embeddings,
                              "clusters/v2/neposredno/testKmeans2.txt", algorithm='kmeans')

clusters.get_clusters_by_word("../data/embeddings/v2/sample_sentences_neposredno.txt", word_json_file, validation_embeddings,
                              "clusters/v2/neposredno/testDbscan.txt", algorithm='dbscan')

clusters.get_clusters_by_word("../data/embeddings/v2/sample_sentences_neposredno.txt", word_json_file, validation_embeddings,
                              "clusters/v2/neposredno/testSpectral.txt", algorithm='spectral')

clusters.get_clusters_by_word("../data/embeddings/v2/sample_sentences_neposredno.txt", word_json_file, validation_embeddings,
                              "clusters/v2/neposredno/testAgglomerativeWard.txt",
                              algorithm='agglomerative', linkage='ward')

clustering.clusters.get_clusters_by_word("../data/embeddings/v2/sample_sentences_neposredno.txt", word_json_file, validation_embeddings,
                                         "clusters/v2/neposredno/testAgglomerativeComplete.txt",
                                         algorithm='agglomerative', linkage='complete')
"""

#clusters.parse_single_cluster_words(word_embeddings, word_json_file, "clusters/v2/single_cluster_words.txt", "clusters/v2/single_cluster_embeddings.txt")

# KMeans - params: n_clusters, n_init
# Spectral - params: n_clusters, separate: distance metric
# Hierarchical - params: n_clusters, linkage, affinity
#   complete linkage requires affinity param
# DBSCAN - params: eps, distance metric

missing_words = ['gora mišic', 'teči kot švicarska ura', 'pleskati se', 'počutiti se kot hrček', 'imunski sistem', 'kovati v nebo', 'kaj se odbije od koga kot od teflona', 'krvna skupina', 'narediti se Francoza', 'palubni potnik', 'kovati v zvezde', 'prvi april', 'menjavati kot gate', 'stresti koga iz hlač', 'jedilna žlica', 'britje norcev', 'črta diskretnosti', 'bliskovita vojna', 'Ariadnina nit', 'suh kot zobotrebec', 'rože ne cvetijo komu', 'plenilski kapitalizem', 'ameriške sanje', 'divja pomaranča', 'boli kot hudič', 'črna oliva', 'zelena magma', 'biti kot živo srebro', 'borza dela', 'družinska srebrnina', 'ne segati niti do gležnjev', 'stric iz ozadja', 'šahovska figura', 'uiti v hlače', 'angleška krona', 'ne hvali dneva pred večerom', 'žvižgalni koncert', 'akcijski radij', 'narediti se francoza', 'trden kot hrast', 'bencin teče komu po žilah', 'hoditi kot pav', 'teta iz ozadja', 'preventiva je boljša kot kurativa', 'hormoni', 'lov na čarovnice', 'bolniški dopust', 'trgati hlače v šolskih klopeh', 'ameriška borovnica', 'metati kladivo', 'čuvati kaj kot zenico svojega očesa', 'belo blago', 'zelena oliva', 'smodniška zarota', 'skromnost je lepa čednost', 'prijokati na svet', 'izprašiti hlače komu', 'točnost je lepa čednost', 'poslednji mohikanec', 'lakmusov test', 'čajna žlica', 'srebro v laseh', 'zaspati kot dojenček', 'prijeti boga za jajca', 'alkoholni maček', 'kolesarski izpit', 'čestitati iz srca', 'bel kot jogurt', 'tolčeni baker', 'jedrska elektrarna', 'igra mačke z mišjo', 'krvni davek', 'delati se Francoza', 'tiščati glavo v pesek kot noj', 'materino mleko', 'podelati se v hlače', 'španska muha', 'sankati se', 'črni ogljik', 'briti norce iz koga', 'čudežna deklica', 'vrtna jagoda', 'deklica za vse', 'kavbojci in Indijanci', 'metati kopje', 'švicarski nož', 'na kredo', 'zaviti v celofan', 'CD-plošča', 'zatišje pred viharjem', 'tresti se kot trepetlika', 'dežno tipalo', 'babilonski stolp', 'radioaktivni jod', 'navadna borovnica', 'alkoholni hlapi', 'padalna obleka', 'obesiti kaj na klin', 'gozdna jagoda', 'desertna žlica', 'med brati povedano', 'enostaven kot pasulj', 'jahalke', 'brusiti nože', 'jesen življenja', 'evro območje', 'brihtna buča', 'žametna revolucija', 'figov list', 'delovati kot švicarska ura', 'trda buča', 'moker kot miš', 'dobiti ošpice', 'delati se francoza', 'ruska ruleta', 'kot cvet se odpirati', 'bled kot kreda', 'potrebovati kaj kot Sahara vodo', 'diabetes tipa 2', 'ameriška brusnica', 'hiteti čemu naproti', 'lepiti se na kaj kot čebele na med', 'bolnišnica za duševne bolezni', 'lakmusov papir', 'gnusiti se', 'biti blažen med ženami', 'veliki brat', 'bel kot kreda', 'lepiti se na koga kot čebele na med', 'beli ribez', 'čili paprika', 'nositi se kot pav', 'ameriški sen', 'ponosen kot pav', 'bas kitara', 'sodobni flamenko', 'brcati žogo', 'arhivsko vino', 'bolnišnica za duševne bolnike', 'bridko je pri srcu komu', 'verižna srajca', 'čudežni deček', 'srce pade komu v hlače', 'bela halja', 'seviljska pomaranča', 'pustiti koga na cedilu', 'čarobna palica', 'brusiti zobe', 'čarobna beseda', 'črn kot saje', 'naglušni', 'kostanjev piknik', 'mandljevo mleko', 'utapljati kaj v alkoholu', 'hitra hoja', 'modra hortenzija', 'tradicionalni flamenko', 'Evansov gambit', 'seči v denarnico', 'čestitati od srca', 'hlače se tresejo komu', 'arabska lutnja', 'verjetnostni račun', 'metati disk', 'dišeča brogovita', 'zavit v celofan', 'siva miš', 'športna plezalka', 'stati kot lipov bog', 'nabasati se', 'koliko je koga v hlačah', 'igra mačke in miši', 'biti kot hrček', 'puli ovratnik', 'ploska noga', 'pri srcu boli koga', 'plosko stopalo', 'sibirski macesen', 'kavboji in Indijanci', 'zaspati na lovorikah', 'sirkova krtača', 'obvezno čtivo', 'cerkveni zbor', 'čestitati s stisnjenimi zobmi', 'ostati na cedilu', 'briti norce iz česa', 'grenka pomaranča', 's srcem v hlačah', 'jagodni izbor', 'erogena cona', 'španska vas', 'živo srebro', 'žoga je okrogla', 'bas bariton', 'človek dialoga', 'japonski lesorez', 'boli kot sam vrag', 'kot zobotrebec', 'jelenova koža', 'hiteti počasi', 'izginiti kot kafra', 'abdominalni oddelek', 'boksati se', 'kavbojci in indijanci', 'kdo nosi hlače', 'zrcalna slika', 'odpreti denarnico', 'raje crkniti, kot ...', 'strici in tete iz ozadja', 'šopiriti se kot pav', 'spati kot dojenček', 'glava boli koga', 'mačji kašelj', 'brusiti pete', 'biti kot brat in sestra', 'verižni oklep', 'kavboji in indijanci', 'kraljev gambit', 'vodilni motiv', 'poslednji Mohikanec', 'sklanjati se', 'postaviti se na trepalnice', 'kaditi kot Turek', 'žepni biljard', 'lasni cilinder']

"""algorithms = [
    {'clusterer': None, 'algorithm': 'kmeans', 'parameters': {'random_state': 42, 'algorithm': 'full'}, 'out_file': None},
    {'clusterer': None, 'algorithm': 'kmeans', 'parameters': {'random_state': 42, 'algorithm': 'elkan'}, 'out_file': None},
]

clusters.find_best_clustering(validation_embeddings, word_json_file, validation_data, "clusters/testAll.txt",
                              algorithms, missing=missing_words)"""

#find_best.find_best_params_main(validation_embeddings, word_json_file, validation_data, "clusters/all", missing=missing_words, use_algorithms=['spectral'])

find_best_v2.find_best_kmeans(validation_embeddings_sample, word_json_file, validation_data, "clusters/kmeans/")
