from data import dataset, gigafida, embeddings
from clustering import find_best

GIVEN_DATA = "data/given_data.txt"
#GIVEN_JSON = "data/given_data.json"
VAL_DATA = "data/labeled_sentences.txt"

#file_helpers.save_json_word_data_from_multiple(GIVEN_DATA, VAL_DATA, GIVEN_JSON)

def dataset_test():
    dataset.create_val_test_set("data/data_2.txt", GIVEN_DATA, "tmp/val.txt", "tmp/test.txt", "tmp/info_val_test.txt", '|')

def sentences_sample_test():
    gigafida.get_sentences_from_gigafida_multiprocess("data/GF", GIVEN_DATA, "tmp/sentences.txt", "tmp/info.txt",
                                                      tmp_dir="tmp/GF", sample_size=1, n_folders=1)

def embeddings_test():
    word_embeddings = embeddings.WordEmbeddings()
    word_embeddings.data_file_to_embeddings(["tmp/sentences.txt"], "tmp/embeddings.txt")

def clustering_test():
    find_best.find_best_kmeans("tmp/embeddings.txt", GIVEN_DATA, VAL_DATA, "tmp")

if __name__ == '__main__':
    dataset_test()
    #sentences_sample_test()
    #embeddings_test()
    #clustering_test()
