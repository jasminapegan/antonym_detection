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
                     "dataset/syn",
                     "dataset/ant") # 20793 lines"""
#dataset.get_bert_embeddings("dataset/ant/train.txt", "dataset/ant/train_embeddings.txt", limit=25)

"""antSynModel = model_old.AntSynModel()
antSynModel.build_model("model/cse_full_model.ckpt")
antSynModel.fit("dataset/dataset_embeddings.txt", range=range(25))
antSynModel.predict("dataset/dataset_embeddings.txt", range=range(25, 30))"""

#antSynModel = model.AntSynModel()
#antSynModel.find_best("dataset/ant/train", "dataset/ant/val", "dataset/ant/out", epochs=5)
#antSynModel.find_best("dataset/syn/train", "dataset/syn/val", "dataset/syn/out", epochs=5)

antSynModel = model.AntSynModel(tokenizer_path="model/saved")
#antSynModel.update_tokenizer(new_tokens=["[BOW]", "[EOW]"])
antSynModel.find_best("dataset/ant/train", "dataset/ant/val", "dataset/ant/out/tokens", epochs=5, mark_word=True)
antSynModel.find_best("dataset/syn/train", "dataset/syn/val", "dataset/syn/out/tokens", epochs=5, mark_word=True)


"""words = {"akuten": "0", "avtomatičen": "0", "bivši": "1", "brat": "0", "dekle": "0", "deklica": "0",
         "deček": "0", "distribucija": "1", "fant": "0", "fant": "1", "fant": "3", "gostitelj": "4",
         "hladen": "6", "izpustiti": "0", "jalov": "3", "kisel": "6", "koncentracija": "2", "koristen": "1",
         "kroničen": "0", "mati": "0", "miren": "0", "mož": "0", "mrtev": "10", "mrtev": "13", "naklonjen": "1",
         "oče": "0", "parazit": "0", "prihodnji": "1", "punca": "2", "ročen": "2", "samec": "0", "samica": "0",
         "sestra": "0", "sladek": "2", "stric": "0", "teta": "0", "tetica": "0", "vzhod": "0", "zahod": "0",
         "zajedalec": "0", "zapreti": "6", "žena": "0", "živ": "8", "živahen": "0", "živahen": "4"}
data.compare_with_new_data("../data/sources/sense/archive/sense_examples_new_multisense.txt", 
                           "../data/sources/sense/sense_examples_new_multisense_2.txt", 
                           words, "dataset/ant_new.txt")"""
