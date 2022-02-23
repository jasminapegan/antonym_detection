import html
import os
from io import TextIOWrapper
import xml.etree.ElementTree as ET
import re
from typing import Dict, Iterable, List

import file_helpers


def write_sense_examples(tree: ET.ElementTree, out_file: TextIOWrapper):
    """
        Parse wordsense xml tree, read examples and write to 'out_file'.

        :param tree: ETree representing wordsense xml file
        :param out_file: examples output file
        :return: None
    """
    root = tree.getroot()

    for entry in root:
        head = entry.find('head')
        headword = head.find('headword')
        lemma = html.unescape(headword.find('lemma').text).strip()
        sense_list = entry.find('body').find('senseList')

        for i, sense in enumerate(sense_list):
            examples = sense.find('exampleContainerList')

            if examples:
                for example in examples:

                    corpusExample = example.find('corpusExample')
                    if corpusExample:

                        sentence = get_text_from_element(corpusExample)
                        headword_occurences = corpusExample.findall("comp[@role='headword']")
                        word_forms = sorted([x.text.strip() for x in headword_occurences])
                        if len(word_forms) == 0:
                            word_form = lemma
                        else:
                            word_form = word_forms[-1]
                            sentence = prepare_tokens(sentence)
                            word_form = prepare_tokens(word_form)

                        try:
                            idx_word = sentence.index(word_form) + 1
                            idx = sentence[:idx_word].count(' ')
                            out_file.write("\t".join([lemma, str(i), str(idx), sentence]) + "\n")
                        except ValueError as e:
                            print("Failed to write for: %s. Error: %s" % (word_form, e))


def write_sense_data(tree: ET.ElementTree, out_file: TextIOWrapper, compound: bool=True):
    """
    Get wordsense data and write it to 'out_file'.

    :param data_file: wordsense xml file
    :param out_file: out data file
    :param compound: consider multiword phrases? (default True)
    :return:
    """

    root = tree.getroot()

    for entry in root:
        head = entry.find('head')
        headword = head.find('headword')

        if not compound and headword.find('lemma').attrib['type'] == 'compound':
            continue

        lemma = html.unescape(headword.find('lemma').text).strip()
        category = "N/A"

        grammar = head.find('grammar')
        if grammar:
            category = grammar.find('category').text

        sense_list = entry.findall('./body/senseList')[0]
        for i, sense in enumerate(sense_list):

            labelList = sense.find("labelList")
            definitionList = sense.find("definitionList")
            definition = sense.find("definitionList/definition[@type='indicator']")

            if definition and definition.text:
                indicator = definition.text
            elif definitionList:
                indicator = get_text_from_element(definitionList, join_str="; ")
            elif labelList:
                indicator = get_text_from_element(labelList, join_str="; ")
            elif len(sense_list) == 1:
                indicator = lemma
            else:
                indicator = "None"
                print("Lemma sense doesn't have definition: " + lemma + ", " + str(i))

            try:
                out_file.write("|".join([str(lemma), str(category), str(i+1), str(indicator)]) + "\n")
            except ValueError as e:
                print("Failed to write for: %s (%s). Error: %s" % (lemma, indicator, e))



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


class WordSense:

    def __init__(self, data_dir, tmp_dir):
        self.data_dir = data_dir
        self.tmp_dir = tmp_dir
        self.clean_dir = os.path.join(self.tmp_dir, "clean")
        self.data_dict = {}

        if not os.path.exists(self.clean_dir):
            os.mkdir(self.clean_dir)

        self.cleanup_files()

    def cleanup_files(self):
        """
        Copies data from files in 'directory' to 'out_dir' skipping unnecessary items.

        :param directory: source directory
        :param out_file: file to write to
        :return: None
        """

        for file in os.listdir(self.data_dir):

            tree = ET.parse(os.path.join(self.data_dir, file))
            self.cleanup_tree(tree)
            tree.write(os.path.join(self.clean_dir, file))

    @staticmethod
    def cleanup_tree(tree: ET.ElementTree):
        """
        Skips unnecessary items.

        - Remove senses with <definition type="indicator"> starting with "stalne zveze",
          "stalna zveza", "frazeološke enote", "frazeologija" ali "fraze"
        - Remove <translationContainerList>

        :param tree: ETree representation of xml file
        :return: None
        """
        root = tree.getroot()

        for parent in root.findall('.//translationContainerList/..'):
            for element in parent.findall('translationContainerList'):
                parent.remove(element)

        regex = re.compile("^([Ss]taln[ae] zvez[ae]|fraze(olo[Šš]ke enote|ologija)?)", re.IGNORECASE)

        for parent in root.findall('.//sense/..'):
            for sense in parent.findall('sense'):
                for definition in sense.findall("./definitionList/definition[@type='indicator']"):
                    indicator = definition.text
                    if indicator and regex.match(indicator.strip()):
                        parent.remove(sense)

    def get_wordsense_examples(self, data_out_file: str, examples_out_file: str):
        """
        Parse wordsense xml 'data_file', find examples or data and write to 'out_file'.

        :param data_out_file: slohun xml file
        :param examples_out_file: examples output file
        :return: None
        """

        for file in os.listdir(self.clean_dir):

            print(file)
            with open(os.path.join(self.clean_dir, file), "r", encoding="utf8") as f:

                tree = ET.parse(f)
                self.get_word_data(tree)

        with open(data_out_file, "w", encoding="utf8") as f:
            with open(examples_out_file, "w", encoding="utf8") as g:
                for word in sorted(list(self.data_dict.keys())):
                    for sense_data in self.data_dict[word].word_sense_list:

                        sense_data.write_to_file(f)

                        for example in sense_data.examples:
                            example.write_to_file(g, word)

    def get_word_data(self, tree: ET.ElementTree):
        root = tree.getroot()
        for entry in root:
            sense = WordSenseDataList(entry)
            word = sense.lemma
            if word not in self.data_dict.keys():
                self.data_dict[word] = sense
            else:
                self.data_dict[word].merge_duplicate_data(sense)

def prepare_tokens(string: str) -> str:
    string = html.unescape(string)
    string = re.sub(r'(?<=[^\s])(?=[.,:;\"\'!?()-])', r' ', string)
    return re.sub(r'(?<=[.,:;\"\'!?()-])(?=[^\s])', r' ', string)

def write_data_to_check(word_data: Dict[str, List[Dict]], out_file: str):
    with open(out_file, "w", encoding="utf8") as f:
        for word in word_data.keys():
            nums = [x['num'] for x in word_data[word]]
            nums_set = set(nums)
            if len(nums) != len(nums_set):
                f.write(str(word_data[word]) + "\n")

def get_text_from_element(element: ET.Element, join_str="") -> str:
    #Returns all the text inside element.
    text = "".join(list(element.itertext())).strip()
    return text.replace("\n", join_str)

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

class Example():

    def __init__(self, word_form, idx, sentence):
        self.word_form = word_form
        self.token_idx = idx
        self.sentence = sentence

    def write_to_file(self, outf, lemma):
        outf.write("\t".join([lemma, self.word_form, str(self.token_idx), self.sentence]) + "\n")

class WordSenseData():
    # specific word sense data including POS tag, sense id, description, examples

    def __init__(self, sense, i, n, lemma, pos_tag):
        self.lemma = lemma
        self.pos_tag = pos_tag
        self.sense_id = i + 1
        self.description = None
        self.examples = []

        self.get_description(sense, n, lemma)

        examples = sense.find('exampleContainerList')
        if examples:
            self.get_examples(examples, lemma)

    def get_description(self, sense, n, lemma):
        labelList = sense.find("labelList")
        definitionList = sense.find("definitionList")
        definition = sense.find("definitionList/definition[@type='indicator']")

        if definition and definition.text:
            self.description = definition.text
        elif definitionList:
            self.description = get_text_from_element(definitionList, join_str="; ")
        elif labelList:
            self.description = get_text_from_element(labelList, join_str="; ")
        elif n == 1:
            self.description = lemma
        else:
            self.description = "None"
            print("Lemma sense doesn't have definition: " + lemma)

    def get_examples(self, examples, lemma):
        for example in examples:
            self.get_example(example, lemma)

    def get_example(self, example, lemma):
        if not example:
            return

        corpusExample = example.find('corpusExample')
        if corpusExample:

            sentence = get_text_from_element(corpusExample)
            headword_occurences = corpusExample.findall("comp[@role='headword']")
            word_forms = sorted([x.text.strip() for x in headword_occurences])

            if len(word_forms) == 0:
                word_form = lemma
            else:
                word_form = word_forms[-1]
                sentence = prepare_tokens(sentence)
                word_form = prepare_tokens(word_form)

            try:
                idx_word = sentence.index(word_form) + 1
                idx = sentence[:idx_word].count(' ')

                self.examples.append(Example(word_form, idx, sentence))
            except Exception as e:
                print(e)

    def write_to_file(self, outf):
        if self.lemma is None or self.pos_tag is None or self.sense_id is None or self.description is None:
            print(self)
        outf.write("|".join([self.lemma, self.pos_tag, str(self.sense_id), self.description]) + "\n")


class WordSenseDataList(object):
    # word sense data list including lemma and sense list

    def __init__(self, entry: ET.Element):
        self.lemma = None
        self.category = "N/A"
        self.word_sense_list = []

        self.parse_head(entry.find('head'))
        self.find_senses(entry.find('.//senseList'))

    def parse_head(self, head):
        headword = head.find('headword')
        self.lemma = html.unescape(headword.find('lemma').text).strip()

        grammar = head.find('grammar')
        if grammar:
            category = grammar.find('category').text
            if category:
                self.category = category

    def find_senses(self, sense_list):
        for i, sense in enumerate(sense_list):
            sense_data = WordSenseData(sense, i, len(sense_list), self.lemma, self.category)
            self.word_sense_list.append(sense_data)

    def merge_duplicate_data(self, sense_list2):
        i = 1
        new_word_data = []

        for d1 in self.word_sense_list:

            diff = [d1.pos_tag != d2.pos_tag or
                    (d1.pos_tag == d2.pos_tag and d1.description != d2.description)
                    for d2 in sense_list2.word_sense_list]

            if all(diff):
                d1.sense_id = i
                new_word_data.append(d1)
                i += 1

        self.word_sense_list = new_word_data
