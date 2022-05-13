import os

from data import embeddings, gigafida, wordsense, dataset, lemmatization
import file_helpers

source_file = "sources/besede_s_pomeni.txt"
lemma_file = "sources/besede_in_leme_s_pomeni.txt"

gigafida_dir = "sources/gigafida/"
out_file = "sentences/sentences.txt"


# KO DOBIM NOVE PODATKE
# 1. pridobi grupirane stavke
#ws = wordsense.WordSense(["sources/wordsense"], "tmp")
#ws.get_wordsense_examples("sources/sense_data.txt", "sources/sense_examples.txt")
#ws.compare_words_data(source_file, "sources/info_words.txt")

#ws = wordsense.WordSense(["sources/wordsense2"], "tmp")
#ws.get_wordsense_examples("sources/sense_data_2.txt", "sources/sense_examples_2.txt")
#ws.compare_words_data(source_file, "sources/info_words_2.txt")

#ws = wordsense.WordSense(["sources/wordsense", "sources/wordsense2"], "tmp/all",
#                         collocations_dir="sources/gf2-collocations/gf2-collocations-extended", clean_data=False)
#ws.get_wordsense_examples("sources/sense_data_new.txt", "sources/sense_examples_new.txt")
#wordsense.compare_words_data(source_file, "sources/info_words_new.txt", ws.examples_file)

# get only multisense words
#file_helpers.get_multisense_words(source_file, "sources/besede_vecpomenske.txt")
#file_helpers.get_multisense_words("sources/sense_data_new.txt", "sources/sense_data_new_multisense.txt")
#file_helpers.filter_file_by_words("sources/sense_examples_new.txt", "sources/sense_data_new_multisense.txt",
#                                  "sources/sense_examples_new_multisense.txt", split_by_2='|')
#wordsense.compare_words_data("sources/besede_vecpomenske.txt", "sources/info_words_new_multisense.txt",
#                             "sources/sense_examples_new_multisense.txt")


# 2. razdeli besede na val in test set
#dataset.create_val_test_set("sources/sense_data_new_multisense.txt","sources/sense_examples_new_multisense.txt",
#                            "sources/besede_vecpomenske.txt", "dataset/val_words_new.txt", "dataset/test_words_new.txt",
#                            "dataset/info_new.txt", '|')

# 3. pridobi sample dodatnih stavkov
#missing_sentences
#file_helpers.filter_file_by_words("sources/sense_data_new_multisense.txt", "sample/gigafida_all_bkp.txt", "sources/new_words_multisense.txt",
#                                  split_by="|", complement=True)

#gigafida.get_sentences_multiprocess(gigafida_dir, "sources/new_words.txt", tmp_dir="tmp/GF",
#                                    folders_range=list(range(0, 100)), sample_size=100, sep="|")
#gigafida.finalize_sentence_search("dataset/gf_all.txt", "sample/gf_all_sample.txt", "sample/gf_all_info.txt",
#                                  tmp_dir="tmp/GF", folders_range=list(range(100)))

#gigafida.get_sentences_multiprocess(gigafida_dir, "dataset/test.txt", tmp_dir="tmp/GF", folders_range=list(range(0, 100)), sep="\t")
#gigafida.finalize_sentence_search("dataset/test.txt","sample/test_sample.txt", "sample/test_info.txt", tmp_dir="tmp/GF", folders_range=list(range(100)))

#file_helpers.concatenate_files(["sample/gigafida_all.txt", "sample/bkp/gigafida_all.txt"], "sample/gigafida_tmp.txt")
#file_helpers.sort_lines("sample/gigafida_tmp.txt", "sample/gigafida_all_new.txt")
#file_helpers.filter_file_by_words("sample/gigafida_all_new.txt", "dataset/val_words_new.txt", "sample/val_new.txt", split_by="\t", split_by_2="|")
#file_helpers.filter_file_by_words("sample/gigafida_all_new.txt", "dataset/test_words_new.txt", "sample/test_new.txt", split_by="\t", split_by_2="|")
#file_helpers.filter_file_by_words("sources/sense_examples_new_multisense.txt", "dataset/val_words_new.txt", "dataset/val_new.txt", split_by_2="|")
#file_helpers.filter_file_by_words("sources/sense_examples_new_multisense.txt", "dataset/test_words_new.txt", "dataset/test_new.txt", split_by_2="|")

# 4. stavki --> embeddings
#we = embeddings.WordEmbeddings()
#we.data_file_to_embeddings(["sample/val_new.txt", "dataset/val_new.txt"], "embeddings/val_embeddings_new.txt", batch_size=1)
#we.data_file_to_embeddings(["sample/test_new.txt", "dataset/test_new.txt"], "embeddings/test_embeddings_new.txt", batch_size=1)

#file_helpers.concatenate_files(["embeddings/bkp/val_embeddings_sorted.txt", "embeddings/bkp/test_embeddings_sorted.txt"], "embeddings/bkp/all.txt")
#file_helpers.filter_file_by_words("embeddings/bkp/all.txt", "dataset/val_words_new.txt", "embeddings/val_old.txt", split_by_2="|")
#file_helpers.filter_file_by_words("embeddings/bkp/all.txt", "dataset/test_words_new.txt", "embeddings/test_old.txt", split_by_2="|")

#file_helpers.concatenate_files(["embeddings/val_embeddings_new.txt", "embeddings/val_old.txt"], "embeddings/all_val.txt")
#file_helpers.concatenate_files(["embeddings/test_embeddings_new.txt", "embeddings/test_old.txt"], "embeddings/all_test.txt")

#file_helpers.sort_lines("embeddings/test_embeddings.txt", "embeddings/test_embeddings_sorted.txt")
#file_helpers.sort_lines("embeddings/val_embeddings.txt", "embeddings/val_embeddings_sorted.txt")

#file_helpers.get_all_words(["dataset/val_words.txt", "dataset/test_words.txt"], "sources/besede_s_pomeni_sorted.txt", "dataset/all_words.txt")



#file_helpers.filter_file_by_words("embeddings/val_embeddings_sorted.txt", "amazonka.txt", "embeddings/amazonka.txt")

#we = embeddings.WordEmbeddings()
#we.data_file_to_embeddings(["sources/sense_examples_new_multisense.txt"], "embeddings/labeled_embeddings.txt", labeled=True, batch_size=1)

import pandas as pd
test = pd.read_csv("tmp/sense_examples.txt", sep="\t", error_bad_lines=False)
print(test['word'][13113])
print(test['word_form'][13113])
print(test['sense_id'][13113])
print(test['word_index'][13113])
print(test['sentence'][13113])

