import os

from data import embeddings, gigafida, embeddings, gigafida, slohun, dataset
import file_helpers

source_file = "sources/besede_s_pomeni.txt"
lemma_file = "sources/besede_in_leme_s_pomeni.txt"
json_words_file = "sources/besede_s_pomeni.json"

# izgleda, da lematizacija ne doprinese veliko
# lemmatization.lemmatize_source(source_file, lemma_file)

#file_helpers.save_json_word_data(source_file, json_words_file)

gigafida_dir = "sources/gigafida/"
out_file = "sentences/sentences.txt"

#gigafida.get_sentences_from_gigafida_multiprocess(gigafida_dir, source_file, out_file)
#print(gigafida.missing_words(source_file, "sentences/"))
#gigafida.get_missing_sentences_from_gigafida(gigafida_dir, "missing_words.txt", "missing_sentences.txt")
# zdaj manjkata le še 'soboljevka' in 'koplanarnost'

#embeddings.get_words_embeddings("embeddings/test.txt")
#print(file_helpers.count_lines("sentences"))
#gigafida.get_sample_sentences(source_file, "sentences", "sample/sample_sentences_100.txt")
#embeddings.get_words_embeddings(["sample/sample_sentences_100.txt"], "embeddings/test.txt", batch_size=1)
#print(file_helpers.file_len("sample/sample_sentences_100.txt"))     # 216169
#print(file_helpers.file_len("embeddings/test.txt"))                 # 158430
#embeddings.get_words_embeddings(["sample/sample_sentences_100.txt"], "embeddings/test2.txt", batch_size=20, skip_i=158430)
#print(file_helpers.file_len("embeddings/sample_sentences_100.txt"))
#file_helpers.sort_lines("embeddings/sample_sentences_100.txt", "embeddings/sample_sentences_100_sorted.txt")


# V2: index of word instead of word form, only long enough sentences or additional sentences

#gigafida.get_sentences_from_gigafida_multiprocess(gigafida_dir, source_file, "sentences/v2/sentences_")
# print(gigafida.missing_words(source_file, "sentences/v2/"))
# ['trebušna slinavka', 'CD-ROM', 'anglikanka', 'CD-predvajalnik', 'kvintdecima', 'trske', 'pobahati se', 'soboljevka', 'sumo', 'koplanarnost', 'balonarka', 'lutnjarka', 'bordarka', 'bajaničarka', 'temenice', 'fantje', 'rumena', 'ekspresionistka', 'ažio', 'hilus', 'gimnastično']
#gigafida.get_missing_sentences_from_gigafida(gigafida_dir, "sentences/v2/missing_words.txt", "sentences/v2/missing_sentences.txt")

#file_helpers.sort_lines("sentences/v2/all_sentences.txt", "sentences/v2/all_sentences_sorted.txt")
#file_helpers.remove_duplicate_lines("sentences/v2/all_sentences_sorted.txt", "sentences/v2/all_sentences_sorted_nondup.txt", range=1)

# preimenovala all_sentences_sorted_nondup.txt v all_sentences.txt
#print(gigafida.missing_words(source_file, "sentences/v2/all_sentences.txt", is_dir=False))
# ['ekspresionistka', 'soboljevka', 'CD-ROM', 'CD-predvajalnik', 'koplanarnost', 'hilus']

#gigafida.get_sample_sentences(source_file, "sentences/v2/all_sentences.txt", "sample/v2/sample_sentences_100.txt")

#embeddings.get_words_embeddings(["sample/v2/sample_sentences_100.txt"], "embeddings/v2/sample_sentences_100.txt", batch_size=1)

#validation.get_slohun_examples("sources/slohun_clarin.xml", "validation/slohun.txt")
#embeddings.get_words_embeddings(["validation/slohun.txt", "validation/akril.txt", "validation/čaj.txt"], "validation/embeddings/validation_dataset.txt", batch_size=1, labeled=True)
#embeddings.get_words_embeddings(["validation/slohun.txt", "validation/akril.txt", "validation/čaj.txt", "validation/neposredno.txt", "validation/jogi.txt"],
#                                      "validation/embeddings/validation_dataset_2.txt", batch_size=1, labeled=True)
#file_helpers.sort_lines("validation/embeddings/validation_dataset.txt", "validation/embeddings/validation_dataset_sorted.txt")


"""
file_helpers.create_sample_word_file("embeddings/v2/sample_sentences_100.txt", "akril",
                                     "embeddings/v2/sample_sentences_akril.txt")
file_helpers.create_sample_word_file("embeddings/v2/sample_sentences_100.txt", "čaj",
                                     "embeddings/v2/sample_sentences_čaj.txt")
file_helpers.create_sample_word_file("embeddings/v2/sample_sentences_100.txt", "neposredno",
                                     "embeddings/v2/sample_sentences_neposredno.txt")
file_helpers.create_sample_word_file("embeddings/v2/sample_sentences_100.txt", "jogi",
                                     "embeddings/v2/sample_sentences_jogi.txt")
"""
#embeddings.get_words_embeddings("embeddings/test.txt")

# Get data from slo-hun dictionary. Interesting data is with more than one cluster and with sentence examples.

#slohun.get_slohun_data("sources/slohun_clarin.xml", "validation/slohun_data_single.txt", compound=False)
#slohun.compare_words_data_2("validation/slohun_single.txt", "sources/besede_s_pomeni.txt")
#gigafida.get_sentences_from_gigafida_multiprocess(gigafida_dir, "sources/slohun_examples_data_minus_given.txt", "sentences/slohun/sentences_")
# print(gigafida.missing_words("sources/slohun_examples_data_minus_given.txt", "sentences/slohun/"))
# 225 manjkajočih
# ['gora mišic', 'teči kot švicarska ura', 'pleskati se', 'počutiti se kot hrček', 'imunski sistem', 'kovati v nebo',
# 'kaj se odbije od koga kot od teflona', 'krvna skupina', 'narediti se Francoza', 'palubni potnik', 'kovati v zvezde',
# 'prvi april', 'menjavati kot gate', 'stresti koga iz hlač', 'jedilna žlica', 'britje norcev', 'črta diskretnosti',
# 'bliskovita vojna', 'Ariadnina nit', 'suh kot zobotrebec', 'rože ne cvetijo komu', 'plenilski kapitalizem',
# 'ameriške sanje', 'divja pomaranča', 'boli kot hudič', 'črna oliva', 'zelena magma', 'biti kot živo srebro',
# 'borza dela', 'družinska srebrnina', 'ne segati niti do gležnjev', 'stric iz ozadja', 'šahovska figura',
# 'uiti v hlače', 'angleška krona', 'ne hvali dneva pred večerom', 'žvižgalni koncert', 'akcijski radij',
# 'narediti se francoza', 'trden kot hrast', 'bencin teče komu po žilah', 'hoditi kot pav', 'teta iz ozadja',
# 'preventiva je boljša kot kurativa', 'hormoni', 'lov na čarovnice', 'bolniški dopust',
# 'trgati hlače v šolskih klopeh', 'ameriška borovnica', 'metati kladivo', 'čuvati kaj kot zenico svojega očesa',
# 'belo blago', 'zelena oliva', 'smodniška zarota', 'skromnost je lepa čednost', 'prijokati na svet',
# 'izprašiti hlače komu', 'točnost je lepa čednost', 'poslednji mohikanec', 'lakmusov test', 'čajna žlica',
# 'srebro v laseh', 'zaspati kot dojenček', 'prijeti boga za jajca', 'alkoholni maček', 'kolesarski izpit',
# 'čestitati iz srca', 'bel kot jogurt', 'tolčeni baker', 'jedrska elektrarna', 'igra mačke z mišjo', 'krvni davek',
# 'delati se Francoza', 'tiščati glavo v pesek kot noj', 'materino mleko', 'podelati se v hlače', 'španska muha',
# 'sankati se', 'črni ogljik', 'briti norce iz koga', 'čudežna deklica', 'vrtna jagoda', 'deklica za vse',
# 'kavbojci in Indijanci', 'metati kopje', 'švicarski nož', 'na kredo', 'zaviti v celofan', 'CD-plošča',
# 'zatišje pred viharjem', 'tresti se kot trepetlika', 'dežno tipalo', 'babilonski stolp', 'radioaktivni jod',
# 'navadna borovnica', 'alkoholni hlapi', 'padalna obleka', 'obesiti kaj na klin', 'gozdna jagoda', 'desertna žlica',
# 'med brati povedano', 'enostaven kot pasulj', 'jahalke', 'brusiti nože', 'jesen življenja', 'evro območje',
# 'brihtna buča', 'žametna revolucija', 'figov list', 'delovati kot švicarska ura', 'trda buča', 'moker kot miš',
# 'dobiti ošpice', 'delati se francoza', 'ruska ruleta', 'kot cvet se odpirati', 'bled kot kreda',
# 'potrebovati kaj kot Sahara vodo', 'diabetes tipa 2', 'ameriška brusnica', 'hiteti čemu naproti',
# 'lepiti se na kaj kot čebele na med', 'bolnišnica za duševne bolezni', 'lakmusov papir', 'gnusiti se',
# 'biti blažen med ženami', 'veliki brat', 'bel kot kreda', 'lepiti se na koga kot čebele na med', 'beli ribez',
# 'čili paprika', 'nositi se kot pav', 'ameriški sen', 'ponosen kot pav', 'bas kitara', 'sodobni flamenko',
# 'brcati žogo', 'arhivsko vino', 'bolnišnica za duševne bolnike', 'bridko je pri srcu komu', 'verižna srajca',
# 'čudežni deček', 'srce pade komu v hlače', 'bela halja', 'seviljska pomaranča', 'pustiti koga na cedilu',
# 'čarobna palica', 'brusiti zobe', 'čarobna beseda', 'črn kot saje', 'naglušni', 'kostanjev piknik', 'mandljevo mleko',
# 'utapljati kaj v alkoholu', 'hitra hoja', 'modra hortenzija', 'tradicionalni flamenko', 'Evansov gambit',
# 'seči v denarnico', 'čestitati od srca', 'hlače se tresejo komu', 'arabska lutnja', 'verjetnostni račun',
# 'metati disk', 'dišeča brogovita', 'zavit v celofan', 'siva miš', 'športna plezalka', 'stati kot lipov bog',
# 'nabasati se', 'koliko je koga v hlačah', 'igra mačke in miši', 'biti kot hrček', 'puli ovratnik', 'ploska noga',
# 'pri srcu boli koga', 'plosko stopalo', 'sibirski macesen', 'kavboji in Indijanci', 'zaspati na lovorikah',
# 'sirkova krtača', 'obvezno čtivo', 'cerkveni zbor', 'čestitati s stisnjenimi zobmi', 'ostati na cedilu',
# 'briti norce iz česa', 'grenka pomaranča', 's srcem v hlačah', 'jagodni izbor', 'erogena cona', 'španska vas',
# 'živo srebro', 'žoga je okrogla', 'bas bariton', 'človek dialoga', 'japonski lesorez', 'boli kot sam vrag',
# 'kot zobotrebec', 'jelenova koža', 'hiteti počasi', 'izginiti kot kafra', 'abdominalni oddelek', 'boksati se',
# 'kavbojci in indijanci', 'kdo nosi hlače', 'zrcalna slika', 'odpreti denarnico', 'raje crkniti, kot ...',
# 'strici in tete iz ozadja', 'šopiriti se kot pav', 'spati kot dojenček', 'glava boli koga', 'mačji kašelj',
# 'brusiti pete', 'biti kot brat in sestra', 'verižni oklep', 'kavboji in indijanci', 'kraljev gambit',
# 'vodilni motiv', 'poslednji Mohikanec', 'sklanjati se', 'postaviti se na trepalnice', 'kaditi kot Turek',
# 'žepni biljard', 'lasni cilinder']

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
