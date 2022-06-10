import csv
import html
import os
import xml.etree.ElementTree as ET
import re
from random import shuffle
from typing import Dict, Iterable, List
import file_helpers
from data.gigafida import get_sentence_by_id, get_now_string


def count_senses(word_count: Dict[str, List[str]], keys: Iterable[str]):
    count = 0
    for key in keys:
        count += len(word_count[key])
    return count

class WordSense:

    def __init__(self, data_dirs, tmp_dir, collocations_dir=None, clean_data=False):
        self.data_dir = data_dirs
        self.tmp_dir = tmp_dir
        self.clean_dir = os.path.join(self.tmp_dir, "clean")
        self.data_dict = {}

        self.collocations_dir = collocations_dir
        self.collocations_dir_clean = os.path.join(self.tmp_dir, "collocations")

        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)

        if not os.path.exists(self.clean_dir):
            os.mkdir(self.clean_dir)

        if not os.path.exists(self.collocations_dir_clean):
            os.mkdir(self.collocations_dir_clean)

        if clean_data:
            self.cleanup_files()
            print("Cleanup done!")

    def cleanup_files(self):
        """
        Copies data from files in 'directory' to 'out_dir' skipping unnecessary items.

        :param directory: source directory
        :param out_file: file to write to
        :return: None
        """

        for d in self.data_dir:
            for file in os.listdir(d):
                print("Cleaning file %s..." % file)

                tree = ET.parse(os.path.join(d, file))
                self.cleanup_tree(tree)

                filename = os.path.basename(file)
                dirname = os.path.basename(d)
                tree.write(os.path.join(self.clean_dir, "%s_%s" % (dirname, filename)))

        """cols_collocations = ['C1_Lemma', 'C2_Lemma', 'C3_Lemma', 'Joint_representative_form_fixed', 'Colocation_ID']
        tmp_clean_dir = os.path.join(self.tmp_dir, "clean_unsorted")

        if not os.path.exists(tmp_clean_dir):
            os.mkdir(tmp_clean_dir)

        for file in os.listdir(self.collocations_dir):

            print("Cleaning file %s..." % file)

            outf = os.path.join(tmp_clean_dir, file)
            filepath = os.path.join(self.collocations_dir, file)
            self.cleanup_csv(filepath, outf, cols_collocations, delimiter=",")

        for file in os.listdir(tmp_clean_dir):
            filepath = os.path.join(tmp_clean_dir, file)
            outf = os.path.join(self.collocations_dir_clean, file)
            file_helpers.write_sorted(filepath, outf, 3)
            file_helpers.file_to_folders(filepath, self.collocations_dir_clean)"""

    @staticmethod
    def cleanup_csv(in_csv, out_csv, cols, delimiter=","):
        with open(in_csv, "r", encoding="utf8") as f:
            with open(out_csv, "w", encoding="utf8") as o:

                reader = csv.DictReader(f, delimiter=delimiter)
                next(reader)  # skip header

                o.write(",".join(cols) + "\n")

                for i, line in enumerate(reader):
                    new_line = delimiter.join([line[x] for x in cols]) + "\n"
                    o.write(new_line)

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

        regex = re.compile("^([Ss]taln[ae] zvez[ae]|fraze(olo[Šš]ke enote|ologija)?|NEUVRŠČENO|WSD)", re.IGNORECASE)

        for parent in root.findall('.//sense/..'):
            for sense in parent.findall('sense'):

                for definition in sense.findall("./definitionList/definition[@type='indicator']"):
                    indicator = definition.text
                    if indicator and regex.match(indicator.strip()):
                        parent.remove(sense)

                for example_container in sense.findall("./exampleContainerList/exampleContainer"):
                    for example in example_container.findall("./corpusExample"):
                        if example.text:
                            example_text = example.text.strip()

                            if example_text.startswith("[ELEXIS-WSD-EN]"):
                                example_container.remove(example)

                            if example_text.startswith("[ELEXIS-WSD-SL]"):
                                example.text = example_text.replace("[ELEXIS-WSD-SL]", "").strip()

    def get_wordsense_examples(self, data_out_file: str, examples_out_file: str):
        """
        Parse wordsense xml 'data_file', find examples or data and write to 'out_file'.

        :param data_out_file: slohun xml file
        :param examples_out_file: examples output file
        :return: None
        """
        self.examples_file = examples_out_file
        self.data_file = data_out_file

        tmp_examples_file = os.path.join(self.tmp_dir, "tmp_examples.txt")
        tmp_data_file = os.path.join(self.tmp_dir, "tmp_data.txt")

        files = os.listdir(self.clean_dir)

        for i, file in enumerate(files):
            print("[%s] %.2f%%: %s" % (get_now_string(), (100*i)/len(files), file))

            lemma_collocations = file in ["WSD-baza-21-04-2022_2,3-pomensko-členjeno.xml",
                                          "WSD-baza-21-04-2022_2,4-pregledani-pomeni.xml",
                                          "WSD-baza-21-04-2022_2-V_DELU.xml"]

            with open(os.path.join(self.clean_dir, file), "r", encoding="utf8") as f:
                tree = ET.parse(f)
                self.get_word_data(tree, lemma_multiword=lemma_collocations)

        print("Finished reading wordsense files")

        with open(tmp_data_file, "w", encoding="utf8") as f:
            with open(tmp_examples_file, "w", encoding="utf8") as g:
                for word in sorted(list(self.data_dict.keys())):
                    for sense_data in self.data_dict[word].word_sense_list:

                        if sense_data.examples:
                            sense_data.write_to_file(f)

                            for example in sense_data.examples:
                                if example:
                                    example.write_to_file(g, sense_data.pos_tag, word, sense_data.sense_id)

        file_helpers.remove_duplicate_lines(tmp_data_file, data_out_file)
        file_helpers.remove_duplicate_lines(tmp_examples_file, examples_out_file)


    def get_word_data(self, tree: ET.ElementTree, lemma_multiword=False):
        root = tree.getroot()
        for i, entry in enumerate(root):
            if i % 10 == 0:
                print("[%s] get_word_data: %d/%d" % (get_now_string(), i, len(root)))
            try:
                sense = WordSenseDataList(entry, lemma_multiword=lemma_multiword)
                word = sense.lemma

                if word not in self.data_dict.keys():
                    self.data_dict[word] = sense
                else:
                    self.data_dict[word].merge_duplicate_data(sense)
            except Exception as e:
                print(f"[{get_now_string()}] Exception: {e}")

def compare_words_data(file_words: str, info_file: str, examples_file: str):
    words_data = [[line[0], line[2]] for line in file_helpers.load_file(file_words, sep='|')]
    words_count = word_list_to_dict(words_data)

    examples_data = []

    with open(examples_file, "r", encoding="utf8") as f:
        for line in f.readlines():
            word, pos_tag, word_form, sense_id, idx, sentence = line.split("\t")
            examples_data += [(word, pos_tag, int(sense_id))]

    examples_data = list(set(examples_data))
    examples_count = word_list_to_dict(examples_data)

    with open(info_file, "w", encoding='utf8') as info:
        words_not_in_examples = [key for key in words_count.keys() if key not in examples_count.keys()]
        examples_not_in_words = [key for key in examples_count.keys() if key not in words_count.keys()]
        intersection = [key for key in examples_count.keys() if key in words_count.keys()]

        info.write("Given words not in examples: %d %d\n" % (len(words_not_in_examples), count_senses(words_count, words_not_in_examples)))
        info.write("Examples not in given words: %d %d\n" % (len(examples_not_in_words), count_senses(examples_count, examples_not_in_words)))
        info.write("Intersection: %d %d\n" % (len(intersection), count_senses(examples_count, intersection)))

        info.write("# given words: %d %d\n" % (len(words_count), count_senses(words_count, words_count.keys())))
        info.write("# example words: %d %d\n" % (len(examples_count), count_senses(examples_count, examples_count.keys())))

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
    text = " ".join(list(element.itertext())).strip()
    re_whitespace = re.compile(r"\s+")
    text = re_whitespace.sub(" ", text)
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

    def write_to_file(self, outf, pos_tag, lemma, sense_id):
        data = [lemma, pos_tag, self.word_form, sense_id, self.token_idx, self.sentence]
        outf.write("\t".join([str(x) for x in data]) + "\n")
        return data

class WordSenseData():
    # specific word sense data including POS tag, sense id, description, examples

    def __init__(self, sense, i, n, lemma, pos_tag, lemma_multiword=False):
        self.lemma = lemma
        self.pos_tag = pos_tag
        self.sense_id = i
        self.description = None
        self.examples = []

        self.get_description(sense, n, lemma)
        if not self.description:
            return

        examples = sense.find('exampleContainerList')
        if examples:
            self.get_examples(examples, lemma, lemma_multiword=lemma_multiword)

    def __repr__(self):
        return f"<lemma: {self.lemma}, pos_tag: {self.pos_tag}, sense_id: {self.sense_id}, description: {self.description}>"

    def get_description(self, sense, n, lemma):
        labelList = sense.find("labelList")
        definitionList = sense.find("definitionList")
        definition = sense.find("definitionList/definition[@type='indicator']")

        if definition and definition.text:
            if definition.text in ["POMEŠANO", "WSD", "SZ", "FE"]:
                return
        if definitionList:
            self.description = get_text_from_element(definitionList, join_str="; ")
        elif labelList:
            self.description = get_text_from_element(labelList, join_str="; ")
        elif n == 1:
            self.description = lemma
        else:
            self.description = "None"
            print("Lemma sense doesn't have definition: " + lemma)

    def get_examples(self, examples, lemma, lemma_multiword=False):
        for i, example in enumerate(examples):
            #print("[%s] Example %d/%d" % (get_now_string(), i, len(examples)))

            if example:
                self.get_example(example, lemma)
                self.get_examples_from_gigafida(example, lemma, lemma_multiword=lemma_multiword)

    def get_example(self, example, lemma):
        corpus_example = example.find('corpusExample')
        if corpus_example:

            sentence = get_text_from_element(corpus_example)
            headword_occurences = corpus_example.findall("comp[@role='headword']")
            word_forms = [x.text.strip() for x in headword_occurences]

            if len(word_forms) == 0:
                word_form = lemma
            else:
                word_form = " ".join(word_forms)
                sentence = prepare_tokens(sentence)
                word_form = prepare_tokens(word_form)

            try:
                if word_form not in sentence:
                    nonimportant = ["se", "me", "te", "ga", "jo", "smo", "ste", "so",
                                    "bom", "boš", "bova", "bomo", "bosta", "boste", "bo", "bodo",
                                    "mi", "mu", "ji", "vas", "nas", "jih"]
                    word_forms = [word_form.replace(x + " ", "") for x in nonimportant] +\
                                 [word_form.replace(" " + x, "") for x in nonimportant]
                    i=0
                    while word_form not in sentence:
                        word_form = word_forms[i]
                        i += 1

                idx_word = sentence.index(word_form) + 1
                idx = sentence[:idx_word].count(' ')

                self.examples.append(Example(word_form, idx, sentence))
            except IndexError:
                pass
                #print("Error: word form '%s' not found in sentence '%s'." % (word_form, sentence))
            except Exception as e:
                pass
                #print("Error: %s. Word: '%s' / sentence: '%s'" % (e, word_form, sentence))

    def get_examples_from_gigafida(self, example, lemma,
                                   gigafida_dir="sources/gigafida",
                                   structures_mapper="sources/strukture.txt",
                                   collocations_dir="tmp/all/collocations",
                                   sentence_mapper_dir="sources/gf2-collocations/gf2-collocation-sentence-mapper",
                                   lemma_multiword=False):

        multiword_example = example.find('multiwordExample')
        if multiword_example:

            collocation = get_text_from_element(multiword_example)

            if 'structureName' not in multiword_example.attrib:
                return

            structure_name = multiword_example.attrib['structureName'].strip()

            if 'structure_id' not in multiword_example.attrib:
                structure_id = find_in_csv_file(structures_mapper, ["structure_name"],
                                                structure_name, "\t", "structure_ID")
            else:
                structure_id = multiword_example.attrib['structure_id'].strip()

            if not structure_id:
                return

            collocations_file = os.path.join(collocations_dir, "%s" % structure_id, lemma[0])

            if not os.path.exists(collocations_file):
                return

            sentence_mapper_file = os.path.join(sentence_mapper_dir, "%s_mapper.txt" % structure_id)

            #if lemma_multiword:
            collocation_parts = collocation.split(" ")
            collocation_parts += (3 - len(collocation_parts)) * [""]
            collocation_id = find_in_csv_file_collocations(collocations_file, [0, 1, 2], #['C1_Lemma', 'C2_Lemma', 'C3_Lemma'],
                                              collocation_parts, ",", 4)
            #else:
            if collocation_id is None:
                collocation_id = find_in_csv_file_collocations(collocations_file, ['Joint_representative_form_fixed'],
                                                       [3], ",", 4) #'Colocation_ID')

            if not collocation_id:
                return

            sentence_ids = find_mapper(sentence_mapper_file, collocation_id)

            if len(sentence_ids) > 10:
                shuffle(sentence_ids)
                sentence_ids = sentence_ids[:10]

            for i, sentence_id in enumerate(sentence_ids):
                res = find_in_gigafida(gigafida_dir, sentence_id)

                if not res:
                    continue

                sentence, lemmas = res

                try:
                    idx_word = lemmas.index(lemma) + 1
                except Exception as e:
                    print(e)
                    continue

                idx = sentence[:idx_word].count(' ')
                word_form = " ".join(sentence.split(" ")[idx: idx + lemma.count(" ") + 1])

                self.examples.append(Example(word_form, idx, sentence))

    def write_to_file(self, outf):
        if self.lemma is None or self.pos_tag is None or self.sense_id is None or self.description is None:
            print(self)

        data = [self.lemma, self.pos_tag, self.sense_id, self.description]
        outf.write("|".join([str(x) for x in data]) + "\n")

        return data

    def save_example(self, word_form, sentence):
        try:
            if word_form not in sentence:
                nonimportant = ["se", "me", "te", "ga", "jo", "smo", "ste", "so",
                                "bom", "boš", "bova", "bomo", "bosta", "boste", "bo", "bodo",
                                "mi", "mu", "ji", "vas", "nas", "jih"]
                word_forms = [word_form.replace(x + " ", "") for x in nonimportant] + \
                             [word_form.replace(" " + x, "") for x in nonimportant]
                i = 0
                while word_form not in sentence:
                    word_form = word_forms[i]
                    i += 1

            idx_word = sentence.index(word_form) + 1
            idx = sentence[:idx_word].count(' ')

            self.examples.append(Example(word_form, idx, sentence))
        except IndexError:
            pass
            # print("Error: word form '%s' not found in sentence '%s'." % (word_form, sentence))
        except Exception as e:
            pass
            # print("Error: %s. Word: '%s' / sentence: '%s'" % (e, word_form, sentence))

def find_in_csv_file(csv_file, cols, value, sep, return_col):
    with open(csv_file, encoding="utf8") as f:

        reader = csv.DictReader(f, delimiter=sep)
        next(reader) # skip header

        for i, line in enumerate(reader):
            col_data = [line[col] for col in cols]

            if col_data == value:
                return line[return_col]
    return None

def find_in_csv_file_collocations(csv_file, cols, value, sep, return_col):
    with open(csv_file, encoding="utf8") as f:

        for i, line in enumerate(f):
            if i == 0:  # skip header
                continue

            line_data = line.strip().split(sep)
            if len(line_data) < 5:
                continue

            col_data = [c for i, c in enumerate(line_data) if i in cols]

            if col_data == value:
                return line_data[return_col]

    return None

def csv_binary_search(reader, a, b, col, value):
    i = (a + b) // 2
    col_data = reader[i][col]

    if value == col_data:
        return reader[i]

    elif value < col_data:
        return csv_binary_search(reader, a, i, col, value)

    else:
        return csv_binary_search(reader, i, b, col, value)

def find_mapper(mapper_file, collocation_id):
    data = []
    found = False

    with open(mapper_file, encoding="utf8") as f:

        for line in f:
            cid, sentence_id = line.split("\t")

            if cid == collocation_id:
                data.append(sentence_id.strip())
                found = True

            elif found:
                return data

    return data

def find_in_gigafida(gigafida_dir, sentence_id):
    folder = "GF%s" % sentence_id[2:4]
    file = "GF%s-dedup.xml" % sentence_id.split('.')[0][2:]
    file_path = os.path.join(gigafida_dir, folder, file)
    return get_sentence_by_id(file_path, sentence_id)

class WordSenseDataList(object):
    # word sense data list including lemma and sense list

    def __init__(self, entry: ET.Element, lemma_multiword=False):
        self.lemma = None
        self.category = "N/A"
        self.word_sense_list = []

        self.parse_head(entry.find('head'))
        self.find_senses(entry.find('.//senseList'), lemma_multiword=lemma_multiword)

    def __repr__(self):
        senses = ", ".join([str(x) for x in self.word_sense_list])
        return f"[{senses}]"

    def parse_head(self, head):
        headword = head.find('headword')
        unescaped = html.unescape(headword.find('lemma').text)
        re_whitespace = re.compile(r"\s+")
        self.lemma = re_whitespace.sub(" ", unescaped).strip()

        grammar = head.find('grammar')
        if grammar:
            category = grammar.find('category').text
            if category:
                self.category = category

    def find_senses(self, sense_list, lemma_multiword=False):
        for i, sense in enumerate(sense_list):
            if i % 10 == 0:
                print("[%s] %s - find_senses: %d/%d" % (get_now_string(), self.lemma, i, len(sense_list)))
            sense_data = WordSenseData(sense, i, len(sense_list), self.lemma, self.category,
                                       lemma_multiword=lemma_multiword)
            if sense_data:
                self.word_sense_list.append(sense_data)

    def merge_duplicate_data(self, sense_list2):
        new_word_data = []
        i = max([x.sense_id for x in self.word_sense_list]) + 1

        for d2 in sense_list2.word_sense_list:
            all_diff = True

            for j, d1 in enumerate(self.word_sense_list):

                if d2.pos_tag == d1.pos_tag and d1.description == d2.description:

                    for example in d2.examples:
                        self.word_sense_list[j].examples.append(example)

                    all_diff = False

            if all_diff:
                d2.sense_id = i
                new_word_data.append(d2)
                i += 1

        self.word_sense_list += new_word_data
