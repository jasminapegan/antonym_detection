import xml.etree.ElementTree as ET
import re
from typing import Dict, Iterable, List

import file_helpers
from data import embeddings


def get_slohun_examples(data_file: str, out_file: str):
    """
    Parse slohun xml 'data_file', read examples and write to 'out_file'.

    :param data_file: slohun xml file
    :param out_file: examples output file
    :return: None
    """

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


def get_slohun_data(data_file: str, out_file: str, compound: bool=True):
    """
    Get slohun data and write it to 'out_file'.

    :param data_file: slohun xml file
    :param out_file: out data file
    :param compound: consider multiword phrases? (default True)
    :return:
    """

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

def compare_words_data(file_slohun_examples: str, file_words: str):
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


def count_senses(word_count: Dict[str, List[str]], keys: Iterable[str]):
    count = 0
    for key in keys:
        count += len(word_count[key])
    return count

def word_list_to_dict(list: List):
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
