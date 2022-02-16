from visualization import visualize

visualize.prepare_data("../clustering/clusters/kmeans/kmeans-algorithm=full,n_init=25,random_state=42",
                       "test/data.tsv", "test/info.tsv")
