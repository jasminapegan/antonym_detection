from data import embeddings, gigafida, slohun, dataset, lemmatization
import file_helpers

source_file = "sources/besede_s_pomeni.txt"
lemma_file = "sources/besede_in_leme_s_pomeni.txt"
json_words_file = "sources/besede_s_pomeni.json"

# izgleda, da lematizacija ne doprinese veliko
# lemmatization.lemmatize_source(source_file, lemma_file)

#file_helpers.save_json_word_data(source_file, json_words_file)

gigafida_dir = "sources/gigafida/"
out_file = "sentences/sentences.txt"

#file_helpers.concatenate_files(["sentences/slohun/sentences_%02d.txt" % i for i in range(100)], "sentences/slohun/all_sentences.txt")
#file_helpers.sort_lines("sentences/slohun/all_sentences.txt", "sentences/slohun/all_sentences_sorted.txt")
#file_helpers.remove_duplicate_lines("sentences/slohun/all_sentences_sorted.txt", "sentences/slohun/all_sentences_sorted_nondup.txt", range=1)
#preimenovala all_sentences_sorted_nondup v all_sentences
#gigafida.get_sample_sentences("sources/slohun_examples_data_minus_given.txt", "sentences/slohun/all_sentences.txt", "sample/slohun/sample_sentences_100.txt")
#embeddings.get_words_embeddings(["sample/slohun/sample_sentences_100.txt"], "embeddings/slohun/sample_sentences_100.txt", batch_size=50)

# create test/val sets
#file_helpers.filter_file_by_words("testing/slohun_dataset.txt", "sources/slohun/slohun_examples_data_minus_given.txt", "testing/slohun_minus_given.txt", skip_idx=1) # 1003
#file_helpers.filter_file_by_words("testing/slohun_dataset.txt", "sources/slohun/slohun_examples_data_minus_given.txt", "testing/slohun_intersect_given.txt", complement=True, skip_idx=1) # 1508
#print(file_helpers.file_len("testing/slohun_minus_given.txt"))
#print(file_helpers.file_len("testing/slohun_intersect_given.txt"))
#file_helpers.get_random_part("testing/slohun_intersect_given.txt", "testing/slohun_intersect_half1.txt", "testing/slohun_intersect_half2.txt", "testing/slohun_half1_words.txt")  # 754/754
#file_helpers.concatenate_files(["testing/slohun_minus_given.txt", "testing/slohun_intersect_half1.txt"], "testing/validation_dataset.txt")
#file_helpers.concatenate_files(["testing/slohun_half1_words.txt", "sources/slohun/slohun_examples_data_minus_given.txt"], "testing/validation_dataset_words.txt")
#file_helpers.filter_file_by_words("embeddings/slohun/sample_sentences_100.txt", "testing/validation_dataset_words.txt", "embeddings/slohun/validation_embeddings.txt")
#file_helpers.filter_file_by_words("embeddings/slohun/sample_sentences_100.txt", "testing/validation_dataset_words.txt", "embeddings/slohun/test_embeddings.txt", complement=True)

#file_helpers.save_json_word_data_from_multiple("sources/besede_s_pomeni.txt", "sources/slohun/slohun_data.txt", "sources/vse_besede.json")
#file_helpers.filter_file_by_words("sources/slohun/slohun.txt", "testing/validation_dataset_words.txt", "sources/slohun/validation_sentences.txt")
#file_helpers.filter_file_by_words("sources/slohun/slohun.txt", "testing/validation_dataset_words.txt", "sources/slohun/test_sentences.txt", complement=True)
#embeddings.get_words_embeddings(["sources/slohun/validation_sentences.txt"], "embeddings/slohun/validation_examples_embeddings.txt")
#embeddings.get_words_embeddings(["sources/slohun/test_sentences.txt"], "embeddings/slohun/test_examples_embeddings.txt")

#file_helpers.concatenate_files(["embeddings/slohun/validation_examples_embeddings.txt", "embeddings/slohun/validation_embeddings.txt"], "testing/validation_dataset_complete.txt")
#file_helpers.concatenate_files(["embeddings/slohun/test_examples_embeddings.txt", "embeddings/slohun/test_embeddings.txt"], "testing/test_dataset_complete.txt")
#file_helpers.sort_lines("testing/validation_dataset_complete.txt", "testing/validation_dataset_complete_sorted.txt")
#file_helpers.sort_lines("testing/test_dataset_complete.txt", "testing/test_dataset_complete_sorted.txt")
# moved complete files to folders

words = []
words_lemmatized = None
words_count = None
gigafida.get_sentences_from_gigafida_multiprocess(gigafida_dir, "sources/slohun/test_data.txt", "sentences/test/sentences.txt", "sentences/test/info.txt", lemmatize=True, sample_size=1)


#gigafida.get_sentences_from_gigafida(gigafida_dir, "sources/slohun/test_data.txt", "sentences/test/sentences.txt", lemmatize=True)


# KO DOBIM NOVE PODATKE
# 1. pridobi grupirane stavke

# 2. razdeli besede na val in test set
# dataset.create_val_test_set("novi podatki", source_file, "val out", "test out")

# 3. pridobi sample dodatnih stavkov
# gigafida.get_sentences_from_gigafida_multiprocess(gigafida_dir, "sources/nova mapa/nekineki", "sentences/nova mapa/neki", lemmatize=True)

# 4. stavki --> embeddings
# embeddings.get_words_embeddings_v2(["dodatni sampli, primeri iz datotek"], "out file")
# file_helpers.filter_file_by_words("vlozitve", "val besede", "vlozitve val")
# file_helpers.filter_file_by_words("vlozitve", "test besede", "vlozitve test")