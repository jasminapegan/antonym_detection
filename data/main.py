from data import embeddings, gigafida, wordsense, dataset, lemmatization
import file_helpers

source_file = "sources/besede_s_pomeni.txt"
lemma_file = "sources/besede_in_leme_s_pomeni.txt"

gigafida_dir = "sources/gigafida/"
out_file = "sentences/sentences.txt"


# KO DOBIM NOVE PODATKE
# 1. pridobi grupirane stavke
#ws = wordsense.WordSense("sources/wordsense", "tmp")
#ws.get_wordsense_examples("sources/sense_data.txt", "sources/sense_examples2.txt")
#ws.compare_words_data(source_file, "sources/info_words2.txt")

# 2. razdeli besede na val in test set
#dataset.create_val_test_set("sources/sense_data.txt","sources/sense_examples.txt", source_file, "dataset/val_words_2.txt",
#                            "dataset/test_words_2.txt", "dataset/info_2.txt", '|')

# 3. pridobi sample dodatnih stavkov

#gigafida.get_sentences_multiprocess(gigafida_dir, "dataset/val.txt", tmp_dir="tmp/GF", folders_range=list(range(0, 100)), sample_size=100, sep="\t")
#gigafida.finalize_sentence_search("dataset/val.txt", "sample/val_sample.txt", "sample/val_info.txt", tmp_dir="tmp/GF", folders_range=list(range(100)))

#gigafida.get_sentences_multiprocess(gigafida_dir, "dataset/test.txt", tmp_dir="tmp/GF", folders_range=list(range(0, 100)), sep="\t")
#gigafida.finalize_sentence_search("dataset/test.txt","sample/test_sample.txt", "sample/test_info.txt", tmp_dir="tmp/GF", folders_range=list(range(100)))

#file_helpers.concatenate_files(["sample/val_sample.txt", "sample/test_sample.txt"], "dataset/gigafida_all.txt")
#file_helpers.filter_file_by_words("dataset/gigafida_all.txt", "dataset/val_words_2.txt", "dataset/val_2.txt", split_by="\t")
#file_helpers.filter_file_by_words("dataset/gigafida_all.txt", "dataset/test_words_2.txt", "dataset/test_2.txt", split_by="\t")


# 4. stavki --> embeddings
#we = embeddings.WordEmbeddings()
#we.data_file_to_embeddings(["sample/val_sample.txt", "dataset/val.txt"], "embeddings/val_embeddings.txt", batch_size=1)
#we.data_file_to_embeddings(["sample/test_sample.txt", "dataset/test.txt"], "embeddings/test_embeddings.txt", batch_size=1)

#file_helpers.sort_lines("embeddings/test_embeddings.txt", "embeddings/test_embeddings_sorted.txt")
#file_helpers.sort_lines("embeddings/val_embeddings.txt", "embeddings/val_embeddings_sorted.txt")

#file_helpers.get_all_words(["dataset/val_words.txt", "dataset/test_words.txt"], "sources/besede_s_pomeni_sorted.txt", "dataset/all_words.txt")



#file_helpers.filter_file_by_words("embeddings/val_embeddings_sorted.txt", "amazonka.txt", "embeddings/amazonka.txt")

