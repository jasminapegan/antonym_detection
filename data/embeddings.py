from datetime import datetime
from io import TextIOWrapper

import classla
import torch
from typing import Iterable, List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BatchEncoding
from file_helpers import load_file


class WordEmbeddings:
    """ A class for work with contextual word embeddings (BERT). """

    Indices = Tuple[int, int]

    def __init__(self, model='EMBEDDIA/crosloengual-bert'):
        """ Initialize lemmatizer, tokenizer and model. """

        classla.download('sl')

        self.lemmatizer = classla.Pipeline('sl', processors='tokenize,pos,lemma', use_gpu=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, output_hidden_states=True)
        self.model.eval()

    def data_file_to_embeddings(self, files: Iterable[str], out_file: str, batch_size: int = 5,
                                labeled: bool = False, lemmatized=True):
        """
        Iterates through 'files', loads data, calculates embeddings and writes them to 'out_file'.

        :param files: a collection of files to be considered
        :param out_file: where to write resulting embeddings tsv data: word, sentence, embedding
        :param batch_size: size of batches of word data to be processed (default 50)
        :param labeled: do the files contain sense labels?
        :return: None
        """

        with open(out_file, "w", encoding="utf8") as outf:

            for f in files:
                print("[%s] opened file %s" % (get_now_string(), f))
                data = load_file(f)
                batches = self.batch(data, batch_size)
                n_batches = len(data) // batch_size

                for i, data_batch in enumerate(batches):
                    if i % 10000 == 0:
                        print("[%s] %d%% (%d / %d)" %
                              (get_now_string(), (100 * i) // n_batches, i, n_batches))
                    #try:
                    self.data_batch_to_embeddings(data_batch, outf, labeled, lemmatized=lemmatized)
                    #except Exception as e:
                    #    print(f"Failed to process batch {i}: {e}")

    def data_batch_to_embeddings(self, data_batch: List, outf: TextIOWrapper, labeled: bool = False, lemmatized=True):
        """
        :param data_batch: batch of data to process
        :param outf: file to write embeddings data
        :param labeled: do the files contain sense labels?
        :return: None
        """
        words, pos_tags, labels, word_indices, sentences = self.parse_data(data_batch, labeled, label_idx=3)
        sentences, dataset, indices = self.prepare_data(words, word_indices, sentences, lemmatized=lemmatized)

        if not sentences:
            return

        with torch.no_grad():
            outputs = self.model(**dataset, output_hidden_states=True)

        if labeled:
            result = self.get_embeddings_from_results(outputs, words, pos_tags, sentences, indices, labels=labels)
        else:
            result = self.get_embeddings_from_results(outputs, words, pos_tags, sentences, indices)


        if result:
            self.write_results_to_file(result, outf)

    def get_words_embeddings(self, words: List[str], pos_tags: List[str], word_indices: List[int], sentences: List[str]) -> List:
        sentences, dataset, indices = self.prepare_data(words, word_indices, sentences)

        with torch.no_grad():
            outputs = self.model(**dataset, output_hidden_states=True)

        return self.get_embeddings_from_results(outputs, words, pos_tags, sentences, indices)

    @staticmethod
    def get_embeddings_from_results(outputs, words: List[str], pos_tags: List[str], sentences: List[str],
                                    indices: List[Indices], labels=[]) -> List:
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
        print(token_embeddings_batch.shape)

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

            if labels:
                result.append([words[i], pos_tags[i], labels[i], sentences[i], cat_vec])
            else:
                result.append([words[i], pos_tags[i], sentences[i], cat_vec])

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

    def prepare_data(self, words: List[str], word_indices: List[int], sentences: List[str], lemmatized=True) \
            -> (List[str], BatchEncoding, List[Indices]):
        """
        Accepts data on usages of a word in sentences and returns sentences where word is switched to its base form,
        a dataset of tokens representing sentences and indices in the sentence between which the word usage is located.

        :param words: list of words senses of which we are observing
        :param word_indices: index of the first word token in corresponding sentence
        :param sentences: sentences where the observed word is used
        :return: tuple (sentences, dataset, indices)
        """

        indices = []
        n = len(word_indices)

        if n == 0:
            return None, None, None

        for i in range(n):
            sentence = sentences[i].split(' ')
            word_split = words[i].split(' ')
            idx = word_indices[i]

            if lemmatized:
                sentence[idx: idx + len(word_split)] = word_split

            sentences[i] = " ".join(sentence)
            new_sentence, word_tokens_range, _ = get_token_range(sentence, words[i], idx, self.tokenizer)

            sentences[i] = new_sentence
            indices.append(word_tokens_range)

        dataset = self.tokenizer([s.split(" ") for s in sentences], padding='longest', return_tensors="pt",
                                 is_split_into_words=True, max_length=512, truncation=True)

        return sentences, dataset, indices

    @staticmethod
    def parse_data(data: List[List], labeled: bool=False, label_idx: int=1) -> (List[str], List, List[int], List[str]):
        """
        Parses word data list and gets its columns: word, labels, indices and sentences.

        :param data: list of data with columns word, labels, indices and sentences
        :param labeled: is the data labeled? (default False)
        :return: list of observed words, list of labels, list of indices of first word token, list of sentences
        """

        words = [d[0] for d in data]
        pos_tags = [d[1] for d in data]
        word_indices = [int(d[-2]) for d in data]
        sentences = [d[-1] for d in data]
        labels = []

        if labeled:
            labels = [d[label_idx] for d in data]

        return words, pos_tags, labels, word_indices, sentences

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

def get_token_range(sentence: List[str], word: str, idx: int, tokenizer) -> (str, WordEmbeddings.Indices):
    """
    Calculates index of first and last token of the word represented by 'word_tokens'.
    """

    sentence_tokens = tokenizer(sentence, is_split_into_words=True)

    words_ids = sentence_tokens.word_ids()
    j, k = 0, len(words_ids) - 1
    word_start = words_ids.index(idx)

    if word_start > 250:
        j = max(0, word_start - 250)
    if k - word_start > 250:
        k = min(word_start + 250, len(words_ids) - 1)

    i1, i2 = sentence_tokens.token_to_word(j), sentence_tokens.token_to_word(k)
    if i1 is None: i1 = 0
    if i2 is None: i2 = len(sentence) - 1
    idx -= i1
    new_sentence = " ".join(sentence[i1: i2 + 1])

    sentence_tokens = tokenizer(new_sentence.split(" "), is_split_into_words=True)
    word_idx_range = (idx, idx + word.count(" "))
    word_tokens_range = (sentence_tokens.word_to_tokens(word_idx_range[0])[0],
                         sentence_tokens.word_to_tokens(word_idx_range[1])[1])

    return new_sentence, word_tokens_range, word_idx_range

