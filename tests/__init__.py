from data import dataset, gigafida

GIVEN_DATA = "data/given_data.txt"

def sanity_test():
    print("Test")

def dataset_test():
    dataset.create_val_test_set("data/labeled_sentences.txt", GIVEN_DATA, "tmp/val.txt", "tmp/test.txt")

def sentences_sample_test():
    gigafida.get_sentences_from_gigafida_multiprocess("data/GF", GIVEN_DATA, "tmp/sentences.txt", "tmp/info.txt",
                                                      lemmatize=True, tmp_dir="tmp/", sample_size=1, n_folders=1)



if __name__ == '__main__':
    sanity_test()
    dataset_test()
    sentences_sample_test()
