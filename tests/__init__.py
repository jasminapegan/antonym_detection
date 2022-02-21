from data import dataset, gigafida, embeddings

GIVEN_DATA = "data/given_data.txt"

def dataset_test():
    dataset.create_val_test_set("data/labeled_sentences.txt", GIVEN_DATA, "tmp/val.txt", "tmp/test.txt")

def sentences_sample_test():
    gigafida.get_sentences_from_gigafida_multiprocess("data/GF", GIVEN_DATA, "tmp/sentences.txt", "tmp/info.txt",
                                                      tmp_dir="tmp/GF", sample_size=1, n_folders=1)

def embeddings_test():
    word_embeddings = embeddings.WordEmbeddings()
    word_embeddings.get_words_embeddings(["tmp/sentences.txt"], "tmp/embeddings.txt")


if __name__ == '__main__':
    dataset_test()
    sentences_sample_test()
    embeddings_test()
