from transformers import AutoTokenizer
from classification.classifier import evaluation

import file_helpers
from classification import model, dataset

ant_syn_folder = "../clustering/ant_syn_senses/archive/sopomenke_protipomenke/"
"""file_helpers.filter_file_by_words("../data/sources/sense/archive/sense_examples_new_multisense.txt",
                                  "../clustering/ant_syn_senses/archive/sopomenke_protipomenke/vse_besede.txt",
                                  "../data/sources/sense/archive/sense_examples_new_multisense_filtered.txt")
file_helpers.fix_indices("../data/sources/sense/archive/sense_examples_new_multisense_filtered.txt",
                         "../data/sources/sense/archive/sense_examples_new_multisense_filtered_fixed.txt", batch_size=250)"""

"""dataset.create_dataset(ant_syn_folder + "napovedi.txt",
                     ant_syn_folder + "ocene_ant.txt",
                     ant_syn_folder + "ocene_syn.txt",
                     "../data/sources/sense/archive/sense_examples_new_multisense_filtered_fixed.txt",
                     "dataset/dataset_syn.txt",
                     "dataset/dataset_ant.txt",
                     "dataset/dataset_anti_syn.txt",
                     "dataset/dataset_anti_ant.txt",
                     "dataset/syn/new2",
                     "dataset/ant/new2",
                     additional_cluster_file="../clustering/ant_syn_senses/description_dist.txt",
                     additional_examples_file="../data/sources/sense/sense_examples_fixed.txt", max_examples=250) # 20793 lines"""
#dataset.get_bert_embeddings("dataset/ant/train.txt", "dataset/ant/train_embeddings.txt", limit=25)"""

"""antSynModel = model_old.AntSynModel()
antSynModel.build_model("model/cse_full_model.ckpt")
antSynModel.fit("dataset/dataset_embeddings.txt", range=range(25))
antSynModel.predict("dataset/dataset_embeddings.txt", range=range(25, 30))"""

#model.find_best("dataset/ant/train", "dataset/ant/val", "out/ant/test", epochs=2, lrs=[1e-5], batch_sizes=[32])
#antSynModel.find_best("dataset/syn/train", "dataset/syn/new/val", "out/syn/new", epochs=5)

#model.find_best("dataset/ant/train", "dataset/ant/val", "dataset/ant/out/tokens", epochs=2, mark_word=True, batch_sizes=[32], syn=False)
#model.find_best("dataset/syn/train", "dataset/syn/val", "dataset/syn/out/tokens", epochs=5, mark_word=True, batch_size=2, resize_model=True)

#tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/crosloengual-bert")
#tokenizer.add_special_tokens({"additional_special_tokens": ["<R1>", "</R1>", "<R2>", "</R2>"]})
#model.score_models("model/ant/best", "dataset/ant/test.txt", "model/ant/test_info.txt", tokenizer, mark_word=True)

#evaluation.score_models("classifier/model/ant/best/dropout_all", "classifier/model/ant/best/dropout_all_test.txt")

#antSynModel.score_models("dataset/ant/out/tokens", "dataset/ant/test.txt", "dataset/ant/test_score.txt")

#model.find_best("dataset/ant/", "dataset/ant/", "out/ant/models", epochs=1, mark_word=True, batch_sizes=[2])
#model.score_models("out/syn/new", "dataset/syn/test.txt", "model/syn/test_info.txt", tokenizer, mark_word=True)

def print_dataset_info(filename):
    w1 = file_helpers.count_words(f"dataset/syn/full/{filename}.txt", indices=[0, 6])
    print(f"Data for {filename}: {len(list(w1.items()))} words")
    for k, v in w1.items():
        print(k, v)
    print()

for i in range(3):
    print_dataset_info(f"train{i}")
    print_dataset_info(f"val{i}")
print_dataset_info("test")
