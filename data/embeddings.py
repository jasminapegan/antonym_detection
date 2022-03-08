from datetime import datetime
from io import TextIOWrapper

import classla
import torch
from typing import Iterable, List, Tuple
from difflib import get_close_matches
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BatchEncoding
from file_helpers import load_file
from data.lemmatization import get_word_lemmas_list
from multiprocessing.pool import ThreadPool as Pool

"""
# replace word form with lemma
def prepare_data(word: str, labels: Iterable, word_indices: Iterable[int], sentences: Iterable[str],
                 tokenizer: AutoTokenizer) -> (List[str], BatchEncoding, List[int], Iterable):
    original_sentences = sentences[:]
    word_split = word.split(" ")

    word_tokens = []
    n = len(labels)

    for i in range(n):
        sentence = sentences[i].split(' ')
        n_words = len(sentence)
        idx = word_indices[i]

        sentence[idx: idx + len(word_split)] = word_split  # some data has incorrect indexing: sentence[idx - len(word): idx] = word
        sentences[i] = " ".join(sentence)
        word_tokens.append(tokenizer.tokenize(word))

        if n_words > 100:
            j = sentence.index(word_split[0])
            sentences[i] = ' '.join(sentence[max(0, j-50): min(n_words, j+50)])

    dataset = tokenizer(sentences, padding='longest', return_tensors="pt", is_split_into_words=False)
    indices = [get_token_range(word_tokens[k], dataset.tokens(k)) for k in range(n)]
    sentences = original_sentences

    overflow_idx = [k for k in range(n) if len(dataset['input_ids'][k]) > 512][::-1]
    if overflow_idx:
        print("overflow idx", overflow_idx)

    for idx in overflow_idx:
        del sentences[idx]
        del dataset[idx]
        del indices[idx]
        del labels[idx]

    return sentences, dataset, indices, labels

def get_token_range(word_tokens, tokens):
    n = len(word_tokens)

    for idx in (i for i, token in enumerate(tokens) if token == word_tokens[0]):
        if tokens[idx: idx+n] == word_tokens:
            return idx, idx + n

    return None

# replace word form with lemma
def get_words_embeddings_v2(files: Iterable[str], out_file: str, batch_size: int=50, skip_i: int=0,
                            labeled: bool=False, pseudoword: bool=False):

    tokenizer = AutoTokenizer.from_pretrained('EMBEDDIA/crosloengual-bert')
    model = AutoModelForSequenceClassification.from_pretrained('EMBEDDIA/crosloengual-bert', output_hidden_states=True)
    model.eval()

    with open(out_file, "w", encoding="utf8") as outf:

        for f in files:
            print(f)
            data = load_file(f)[skip_i:]

            for i, data_batch in enumerate(batch(data, batch_size)):

                if i % 100 == 0:
                    print(i, "/", len(data) // batch_size)

                word, labels, word_indices, sentences = get_words_labels_indices_sentences(data_batch, labeled)
                sentences, dataset, indices, labels = prepare_data(word, labels, word_indices, sentences, tokenizer)

                with torch.no_grad():
                    outputs = model(**dataset, output_hidden_states=True)

                if pseudoword:
                    result = get_psewdoword_embedding_from_results(outputs, labels, word, sentences, indices, labeled)
                else:
                    result = get_embeddings_from_results(outputs, labels, word, sentences, indices, labeled)

                write_results_to_file(result, outf, labeled)


def write_results_to_file(result, outf):
    for line in result:
        embedding = " ".join([str(float(x)) for x in line[-1]])
        line[-1] = embedding
        outf.write("\t".join(line) + "\n")


def get_psewdoword_embedding_from_results(outputs, word, sentences, indices):
    raise NotImplementedError("Pseudoword embeddings are not implemented")


def get_embeddings_from_results(outputs, word: str, sentences: List[str], indices: ):
    # hidden states is 4dim: layers / batches / tokens / features (0, 1, 2, 3)
    hidden_states = outputs.hidden_states
    token_embeddings_batch = torch.stack(hidden_states, dim=0)

    # order we want: batches / tokens / layers / features (1, 2, 0, 3)
    token_embeddings_batch = token_embeddings_batch.permute(1, 2, 0, 3)

    result = []

    for i, token_embeddings in enumerate(token_embeddings_batch):
        idx_from, idx_to = indices[i]
        token_embeddings = token_embeddings[idx_from: idx_to, -4:]
        size, dimension = token_embeddings.size(), token_embeddings.dim()

        if size[0] > 1:
            lay4, lay3, lay2, lay1 = torch.mean(token_embeddings, dim=0, keepdim=False)

        else:
            lay4, lay3, lay2, lay1 = token_embeddings[0]

        cat_vec = torch.cat((lay4, lay3, lay2, lay1), dim=0)

        result.append([word, sentences[i], cat_vec])

    return result


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_words_labels_indices_sentences(data, labeled):
    word = data[0][0]
    labels = []

    if labeled:
        labels = [int(d[1]) for d in data]

    word_indices = [int(d[-2]) for d in data]
    sentences = [d[-1] for d in data]
    return word, labels, word_indices, sentences
"""


class WordEmbeddings:
    """ A class for work with contextual word embeddings (BERT). """

    Indices = Tuple[int, int]

    def __init__(self, model='EMBEDDIA/crosloengual-bert'):
        """ Initialize lemmatizer, tokenizer and model. """

        self.lemmatizer = classla.Pipeline('sl', processors='tokenize,pos,lemma', use_gpu=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, output_hidden_states=True)
        self.model.eval()

    def get_words_embeddings(self, word: str, sentences: List[str]) -> (List, List[int]):
        """
        Given a word and a list of sentences containing this word, calculates embeddings of the word in each sentence.

        :param word: the observed word
        :param sentences: a list of sentences where observed word is used
        :return: a list of embeddings and a list of skipped indices due to errors
        """

        word_lemma = get_word_lemmas_list([word], lemmatizer=self.lemmatizer)[0]
        sentences_lemmatized = get_word_lemmas_list(sentences[:], lemmatizer=self.lemmatizer)
        words = len(sentences) * [word]
        skipped_idx = []

        for i in range(len(sentences_lemmatized)):
            s = sentences_lemmatized[i]

            if word_lemma not in s:
                matches = get_close_matches(word_lemma, s.split(' '), n=1)

                if len(matches) == 0:
                    skipped_idx.append(i)
                else:
                    sentences_lemmatized[i] = s.replace(matches[0], word_lemma)

        indices = [s[:s.index(word_lemma) + 1].count(' ') for s in sentences_lemmatized if word_lemma in s]

        if len(indices) == 0:
            return []

        sentences, dataset, indices = self.prepare_data(words, indices, sentences)

        with torch.no_grad():
            outputs = self.model(**dataset, output_hidden_states=True)

        results = self.get_embeddings_from_results(outputs, words, sentences, indices)

        return [r[-1] for r in results], skipped_idx


    def data_file_to_embeddings(self, files: Iterable[str], out_file: str, batch_size: int = 50, labeled: bool = False,
                                pseudoword: bool = False):
        """
        Iterates through 'files', loads data, calculates embeddings and writes them to 'out_file'.

        :param files: a collection of files to be considered
        :param out_file: where to write resulting embeddings tsv data: word, sentence, embedding
        :param batch_size: size of batches of word data to be processed (default 50)
        :param labeled: do the files contain sense labels?
        :param pseudoword: use pseudoword embeddings? (default False)
        :return: None
        """

        with open(out_file, "w", encoding="utf8") as outf:

            for f in files:
                print("[%s] opened file %s" % (get_now_string(), f))
                data = load_file(f)
                batches = self.batch(data, batch_size)

                for i, data_batch in enumerate(batches):
                    if i % 100 == 0:
                        print("[%s] %d / %d" % (get_now_string(), i, len(data) // batch_size))
                    self.data_batch_to_embeddings(data_batch, outf, labeled, pseudoword)

    def data_batch_to_embeddings(self, data_batch: List, outf: TextIOWrapper, labeled: bool = False,
                                 pseudoword: bool = False):
        """
        :param data_batch: batch of data to process
        :param outf: file to write embeddings data
        :param labeled: do the files contain sense labels?
        :param pseudoword: use pseudoword embeddings? (default False)
        :return: None
        """
        words, labels, word_indices, sentences = self.parse_data(data_batch, labeled)
        sentences, dataset, indices = self.prepare_data(words, word_indices, sentences)

        with torch.no_grad():
            outputs = self.model(**dataset, output_hidden_states=True)

        if pseudoword:
            raise NotImplementedError("Pseudoword embeddings not implemented")
            # result = get_psewdoword_embedding_from_results(outputs, word, sentences, indices)
        else:
            result = self.get_embeddings_from_results(outputs, words, sentences, indices)

        self.write_results_to_file(result, outf)

    def get_words_embeddings_2(self, words: List[str], word_indices: List[int], sentences: List[str],
                               pseudoword: bool=False) -> List:
        sentences, dataset, indices = self.prepare_data(words, word_indices, sentences)

        with torch.no_grad():
            outputs = self.model(**dataset, output_hidden_states=True)

        if pseudoword:
            raise NotImplementedError("Pseudoword embeddings not implemented")
            # result = get_psewdoword_embedding_from_results(outputs, word, sentences, indices)
        else:
            return self.get_embeddings_from_results(outputs, words, sentences, indices)

    @staticmethod
    def get_embeddings_from_results(outputs, words: List[str], sentences: List[str], indices: List[Indices]) -> List:
        """
        Manipulates results from model to get word embeddings. The embeddings are calculated as mean of concatenated
        last 4 layers representing word tokens.

        :param outputs: outputs of embedding model
        :param words: observed words
        :param sentences: a list of sentences containing word
        :param indices: index pairs representing indices of first and last token of the word in tokenized sentence
        :return: a list of lines containing word, sentence and contextual embedding of the word
        """

        # hidden states is 4dim: layers / batches / tokens / features (0, 1, 2, 3)
        hidden_states = outputs.hidden_states
        token_embeddings_batch = torch.stack(hidden_states, dim=0)

        # order we want: batches / tokens / layers / features (1, 2, 0, 3)
        token_embeddings_batch = token_embeddings_batch.permute(1, 2, 0, 3)

        result = []

        for i, token_embeddings in enumerate(token_embeddings_batch):
            idx_from, idx_to = indices[i]
            token_embeddings = token_embeddings[idx_from: idx_to, -4:]
            size, dimension = token_embeddings.size(), token_embeddings.dim()

            if size[0] > 1:
                lay4, lay3, lay2, lay1 = torch.mean(token_embeddings, dim=0, keepdim=False)

            else:
                lay4, lay3, lay2, lay1 = token_embeddings[0]

            cat_vec = torch.cat((lay4, lay3, lay2, lay1), dim=0)

            result.append([words[i], sentences[i], cat_vec])

        return result

    @staticmethod
    def write_results_to_file(result: List, outf: TextIOWrapper):
        """
        Writes a list of results into opened file.

        :param result: result list
        :param outf: file wrapper to write data to output file
        :return: None
        """

        for line in result:
            embedding = " ".join([str(float(x)) for x in line[-1]])
            line[-1] = embedding
            outf.write("\t".join(line) + "\n")


    def prepare_data(self, words: List[str], word_indices: List[int], sentences: List[str]) \
            -> (List[str], BatchEncoding, List[Indices]):
        """
        Accepts data on usages of a word in sentences and returns sentences where word is switched to its base form,
        a dataset of tokens representing sentences and indices in the sentence between which the word usage is located.

        :param words: list of words senses of which we are observing
        :param word_indices: index of the first word token in corresponding sentence
        :param sentences: sentences where the observed word is used
        :return: tuple (sentences, dataset, indices)
        """

        original_sentences = sentences[:]
        word_tokens = []
        n = len(word_indices)

        for i in range(n):
            sentence = sentences[i].split(' ')
            word_split = words[i].split(' ')
            n_words = len(sentence)
            idx = word_indices[i]


            sentence[idx: idx + len(word_split)] = word_split  # some data has incorrect indexing: sentence[idx - len(word): idx] = word

            sentences[i] = " ".join(sentence)
            word_tokens.append(self.tokenizer.tokenize(words[i]))

            if n_words > 100:
                j = sentence.index(word_split[0])
                sentences[i] = ' '.join(sentence[max(0, j - 50): min(n_words, j + 50)])

        dataset = self.tokenizer(sentences, padding='longest', return_tensors="pt", is_split_into_words=False)
        indices = [self.get_token_range(word_tokens[k], dataset.tokens(k)) for k in range(n)]
        sentences = original_sentences

        missing_indices = [str(i) for i in range(n) if indices[i] == (-1, -1)]
        if missing_indices:
            print("Missing indices: " + ",".join(missing_indices))

            for idx in missing_indices:
                del sentences[int(idx)]
                del dataset[int(idx)]
                del indices[int(idx)]

        #assert len(missing_indices) == 0, "Missing indices: " + ",".join(missing_indices)

        for k in range(n - len(missing_indices)):
            dataset['input_ids'][k] = self.trim_index(indices[k][1], dataset['input_ids'][k])

        return sentences, dataset, indices

    @staticmethod
    def trim_index(idx_to: int, tokens: List, limit: int=512) -> List:
        """
        Returns trimmed token list, with regards to index of the word we want to keep in resulting token list.

        :param idx_to: index of the last token we want to keep in resulting token list
        :param tokens: list of tokens we want trimmed
        :param limit: how many tokens to trim to (default 512)
        :return: trimmed token list
        """

        length = len(tokens)

        if length > limit:
            diff = length - limit

            if idx_to < 512:
                return tokens[:limit]
            else:
                return tokens[diff:]

        return tokens

    @staticmethod
    def get_token_range(word_tokens: List, tokens: List) -> Indices:
        """
        Calculates index of first and last token of the word represented by 'word_tokens'.

        :param word_tokens: a list of tokens representing observed word
        :param tokens: a list of tokens representing observed word in context, ex. a sentence
        :return: a tuple containing index of the first word token and index of the last word token in given token list.
        """

        n = len(word_tokens)

        for idx in (i for i, token in enumerate(tokens) if token == word_tokens[0]):
            if tokens[idx: idx+n] == word_tokens:
                return idx, idx + n

        return -1, -1

    @staticmethod
    def parse_data(data: List[List], labeled: bool=False) -> (List[str], List, List[int], List[str]):
        """
        Parses word data list and gets its columns: word, labels, indices and sentences.

        :param data: list of data with columns word, labels, indices and sentences
        :param labeled: is the data labeled? (default False)
        :return: list of observed words, list of labels, list of indices of first word token, list of sentences
        """

        words = [d[0] for d in data]
        word_indices = [int(d[-2]) for d in data]
        sentences = [d[-1] for d in data]
        labels = []

        if labeled:
            labels = [int(d[1]) for d in data]

        return words, labels, word_indices, sentences

    @staticmethod
    def batch(iterable: List, n: int=1):
        """
        Using 'iterable' generates batches of size 'n'.

        :param iterable: a list of which batches we need
        :param n: batch size
        :return: generator of batches
        """

        length = len(iterable)
        for ndx in range(0, length, n):
            yield iterable[ndx:min(ndx + n, length)]

def get_now_string():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")