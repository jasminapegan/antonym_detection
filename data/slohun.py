import xml.etree.ElementTree as ET
import re

import file_helpers


def get_slohun_examples(data_file, out_file):
    with open(out_file, "w", encoding="utf8") as f:

        tree = ET.parse(data_file)
        root = tree.getroot()

        for entry in root:
            head = entry.find('head')
            headword = head.find('headword')
            lemma = headword.find('lemma').text
            sense_list = entry.find('body').find('senseList')

            for i, sense in enumerate(sense_list):

                examples = sense.find('exampleContainerList')
                if examples:
                    for example in examples:

                        corpusExample = example.find('corpusExample')
                        if corpusExample:

                            sentence = ""
                            if corpusExample.text:
                                sentence = corpusExample.text

                            headword_occurences = corpusExample.findall('comp')

                            for headword_occurence in headword_occurences:
                                word_form = headword_occurence.text
                                sentence += word_form

                                if headword_occurence.tail:
                                    sentence += headword_occurence.tail

                            sentence = re.sub(r'(?<=[^\s])(?=[.,:;\"\'!?()-])', r' ', sentence)
                            sentence = re.sub(r'(?<=[.,:;\"\'!?()-])(?=[^\s])', r' ', sentence)

                            try:
                                idx = sentence.split(" ").index(word_form)
                                f.write("\t".join([lemma, str(i), str(idx), sentence]) + "\n")
                            except ValueError:
                                print("Failed to write for: " + word_form)


def get_slohun_data(data_file, out_file, compound=True):
    with open(out_file, "w", encoding="utf8") as f:

        tree = ET.parse(data_file)
        root = tree.getroot()

        for entry in root:
            head = entry.find('head')
            headword = head.find('headword')

            if not compound and headword.find('lemma').attrib['type'] == 'compound':
                continue

            lemma = headword.find('lemma').text

            category = "None"
            grammar = head.find('grammar')
            if grammar:
                category = grammar.find('category').text

            sense_list = entry.find('body').find('senseList')

            for i, sense in enumerate(sense_list):
                definition_list = sense.find('definitionList')
                if definition_list is None:
                    print("Lemma sense doesnt have definition: " + lemma + ", " + str(i))
                indicator = sense.find('definitionList')[0].text

                try:
                    f.write("|".join([lemma, category, str(i+1), indicator]) + "\n")
                except ValueError:
                    print("Failed to write for: %s (%s)" % (lemma, indicator))


def compare_words_data(file_slohun, file_words):
    words_data = file_helpers.load_file(file_words, sep='|')
    slohun_data = file_helpers.load_file(file_slohun, sep='|')

    words_not_in_slohun = [line for line in words_data if line not in slohun_data]
    slohun_not_in_words = [line for line in slohun_data if line not in words_data]
    intersection = [line for line in slohun_data if line in words_data]
    print("Words not in slohun:", len(words_not_in_slohun))
    print("Slohun not in words:", len(slohun_not_in_words))
    print("Intersection:", len(intersection))

    words_dict = word_list_to_dict(words_data)
    slohun_dict = word_list_to_dict(slohun_data)

    count = 0
    for key in words_dict.keys():
        if key in slohun_dict.keys():
            if len(words_dict[key]) != len(slohun_dict[key]):
                print(key)
                print(words_dict[key])
                print(slohun_dict[key])
                count += 1

    print("# different entries:", count)


def compare_words_data_2(file_slohun_examples, file_words):
    words_data = [[line[0], line[2]] for line in file_helpers.load_file(file_words, sep='|')]
    words_count = word_list_to_dict(words_data)

    slohun_data = []

    with open(file_slohun_examples, "r", encoding="utf8") as f:
        for line in f.readlines():
            word, sense_id, idx, sentence = line.split("\t")
            slohun_data += [(word, int(sense_id))]

    slohun_data = set(slohun_data)
    slohun_count = word_list_to_dict(list(slohun_data))

    words_not_in_slohun = [key for key in words_count.keys() if key not in slohun_count.keys()]
    slohun_not_in_words = [key for key in slohun_count.keys() if key not in words_count.keys()]
    intersection = [key for key in slohun_count.keys() if key in words_count.keys()]
    print("Words not in slohun:", len(words_not_in_slohun), count_senses(words_count, words_not_in_slohun))
    print("Slohun not in words:", len(slohun_not_in_words), count_senses(slohun_count, slohun_not_in_words))
    print("Intersection:", len(intersection), count_senses(slohun_count, intersection))


def count_senses(word_count, keys):
    count = 0
    for key in keys:
        count += len(word_count[key])
    return count


def word_list_to_dict(list):
    list.sort(key=lambda x: x[0])

    words_dict = {}
    for line in list:

        word = line[0]
        indicator = line[-1]

        if word in words_dict.keys():
            words_dict[word].append(indicator)
        else:
            words_dict[word] = [indicator]

    return words_dict


def create_test_val_dataset(given_words_file, sentences_file, tmp_dir, out_dir):
    sentences_minus_given =  tmp_dir + "/slohun_sentences_minus_given.txt"
    sentences_intersect_given =  tmp_dir + "/slohun_sentences_intersect_given.txt"
    sentences_intersect_given_1 =  tmp_dir + "/slohun_sentences_intersect_given_1.txt"
    sentences_intersect_given_2 =  tmp_dir + "/slohun_sentences_intersect_given_2.txt"
    words_intersect_given_1 =  tmp_dir + "/slohun_words_intersect_given_1.txt"
    validation_dataset_unsorted = tmp_dir + "/validation_dataset.txt"
    test_dataset_unsorted = tmp_dir + "/validation_dataset.txt"

    validation_dataset = out_dir + "/validation_dataset.txt"
    test_dataset = out_dir + "/validation_dataset.txt"
    validation_words = out_dir + "/validation_words.txt"
    test_words = out_dir + "/test_words.txt"

    file_helpers.filter_file_by_words(sentences_file, given_words_file, sentences_minus_given, skip_idx=1, complement=True)
    file_helpers.filter_file_by_words(sentences_file, given_words_file, sentences_intersect_given, skip_idx=1, complement=False)
    file_helpers.get_random_part(sentences_intersect_given, sentences_intersect_given_1, sentences_intersect_given_2, words_intersect_given_1)

    embeddings_v2.get_words_embeddings_v2([sentences_minus_given, sentences_intersect_given_1], validation_dataset_unsorted, batch_size=100)
    embeddings_v2.get_words_embeddings_v2([sentences_intersect_given_2], test_dataset_unsorted, batch_size=100)

    file_helpers.sort_lines(validation_dataset_unsorted, validation_dataset)
    file_helpers.sort_lines(test_dataset_unsorted, test_dataset)

    file_helpers.get_unique_words(validation_dataset, validation_words, sep='\t')
    file_helpers.get_unique_words(test_dataset, test_words, sep='\t')
