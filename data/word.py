""" Works only with files sorted by words. """
from file_helpers import convert_to_np_array, file_len
from typing import List, Dict, Iterator
import numpy as np


class WordData:

    def __init__(self, data: List[List[str]]):
        """
        Initialize WordData object containing word, usages in sentences, their embeddings.

        :param data: A list of lines (string lists) representing word data.
        """

        self.words = [d[0] for d in data]
        assert len(set(self.words)) <= 1

        self.word = self.words[0]

        self.sentences = [d[-2].strip() for d in data]
        self.embeddings = convert_to_np_array([d[-1] for d in data])

        self.n_sentences = len(self.sentences)
        self.val_ids = None
        self.predicted_labels = None
        self.validation_labels = None


    def add_missing_sentences(self, missing_sentences: List[str], word_embeddings, word_val_data: Dict):
        words = len(missing_sentences) * [self.word]
        indices = [i for i, s in zip(word_val_data['indices'], word_val_data['sentences']) if s in missing_sentences]
        results = word_embeddings.get_words_embeddings_2(words, indices, missing_sentences[:])
        missing_embeddings = [r[-1] for r in results]

        self.sentences += missing_sentences
        missing_embeddings = np.array([np.array(x, dtype=float) for x in missing_embeddings])
        self.embeddings = np.append(self.embeddings, missing_embeddings, axis=0)

    def set_predicted_labels(self, labels):
        self.predicted_labels = [labels[i] for i in self.val_ids]

        if (len(self.predicted_labels) != len(self.validation_labels)):
            print("Labels not matching in length!")

def word_data_gen(file_path: str, progress: int=None) -> Iterator[WordData]:
    """
    Generates word data read from sorted tsv file at 'file_path'.

    :param file_path: tsv file containing word data, sorted by first field (word)
    :param progress: print progress every 'progress' lines. If None, do not print progress (default: None)
    :return: iterator over words - yielding WordData on selected word
    """

    data = []
    word = None
    n_lines = 0

    if progress:
        n_lines = file_len(file_path)

    with open(file_path, "r", encoding="utf8") as f:
        for i, line in enumerate(f):

            if progress and i % progress == 0:
                print("Word data progress: %d / %d" % (i, n_lines))

            data_line = line.strip().split('\t')
            data_word = data_line[0]

            # finished reading word data
            if data_word != word and word != None and len(data) > 0:

                if len(data) > 0:
                    yield WordData(data)
                    word = data_word
                    data = []

            else:
                data.append(data_line)
                word = data_word

    yield WordData(data + [data_line])
