from transformers import AutoTokenizer

import file_helpers
from classification import model, dataset, scoring

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
                     "dataset/syn/new",
                     "dataset/ant/new",
                     additional_cluster_file="../clustering/ant_syn_senses/description_dist.txt",
                     additional_examples_file="../data/sources/sense/sense_examples_fixed.txt", max_examples=8000) # 20793 lines"""
#dataset.get_bert_embeddings("dataset/ant/train.txt", "dataset/ant/train_embeddings.txt", limit=25)

"""antSynModel = model_old.AntSynModel()
antSynModel.build_model("model/cse_full_model.ckpt")
antSynModel.fit("dataset/dataset_embeddings.txt", range=range(25))
antSynModel.predict("dataset/dataset_embeddings.txt", range=range(25, 30))"""

#model.find_best("dataset/ant/train", "dataset/ant/val", "out/ant/test", epochs=2, lrs=[1e-5], batch_sizes=[32])
#antSynModel.find_best("dataset/syn/train", "dataset/syn/new/val", "out/syn/new", epochs=5)

tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/crosloengual-bert")
model.update_tokenizer(tokenizer, new_tokens=["[BASE]", "[SYN]", "[ANT]"], out_dir="model/saved")
model.find_best("dataset/ant/train", "dataset/ant/val", "dataset/ant/out/tokens", epochs=1, mark_word=True, batch_sizes=[32], resize_model=True, tokenizer_path="model/saved", syn=False)
#model.find_best("dataset/syn/train", "dataset/syn/val", "dataset/syn/out/tokens", epochs=5, mark_word=True, batch_size=2, resize_model=True)

#antSynModel.score_models("dataset/ant/out/tokens", "dataset/ant/test.txt", "dataset/ant/test_score.txt")
