""" Works only with files sorted by words. """
from difflib import get_close_matches
from typing import List, Dict

import numpy as np
import torch

from data.embeddings import get_embeddings_from_results, prepare_data
from file_helpers import convert_to_np_array, file_len


def word_data_gen(file_path: str, progress=None):
    data = []
    word = None

    if progress:
        n_lines = file_len(file_path)

    with open(file_path, "r", encoding="utf8") as f:
        for i, line in enumerate(f):

            if progress and i % progress == 0:
                print("Word data progress: %d / %d" % (i, n_lines))

            data_line = line.strip().split("\t")
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

    if len(data) > 0:
        yield WordData(data)


class WordData:

    def __init__(self, data):
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
        print(missing_sentences)
        indices = [i for i, s in zip(word_val_data['indices'], word_val_data['sentences']) if s in missing_sentences]
        sentences, dataset, _, labels = prepare_data(self.word, [], indices, missing_sentences, word_embeddings.tokenizer)

        n_words = len(self.word.split(' '))
        indices_from_to = [(i, i + n_words) for i in indices]

        with torch.no_grad():
            outputs = word_embeddings.model(**dataset, output_hidden_states=True)

        results = get_embeddings_from_results(outputs, labels, self.word, sentences, indices_from_to, False)
        missing_embeddings = [r[-1] for r in results]

        self.sentences += missing_sentences
        missing_embeddings = np.array([np.array(x, dtype=float) for x in missing_embeddings])
        self.embeddings = np.append(self.embeddings, missing_embeddings, axis=0)

    def set_predicted_labels(self, labels):
        self.predicted_labels = [labels[i] for i in self.val_ids]

        if (len(self.predicted_labels) != len(self.validation_labels)):
            print("Labels not matching in length!")

