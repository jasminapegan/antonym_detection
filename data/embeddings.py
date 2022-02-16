import classla
import torch
from difflib import get_close_matches
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from file_helpers import load_file
from data.lemmatization import get_word_lemmas_list


def get_token_range(word_tokens, tokens):
    n = len(word_tokens)

    for idx in (i for i, token in enumerate(tokens) if token == word_tokens[0]):
        if tokens[idx: idx+n] == word_tokens:
            return idx, idx + n

    return None


# replace word form with lemma
def prepare_data(word, labels, word_indices, sentences, tokenizer):
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


# replace word form with lemma
def get_words_embeddings_v2(files, out_file, batch_size=50, skip_i=0, labeled=False, pseudoword=False):
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


def get_psewdoword_embedding_from_results(outputs, labels, word, sentences, indices, labeled):
    pass


def get_embeddings_from_results(outputs, labels, word, sentences, indices, labeled):
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

        if labeled:
            result.append([word, str(labels[i]), sentences[i], cat_vec])
        else:
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


class WordEmbeddings:

    def __init__(self):

        self.lemmatizer = classla.Pipeline('sl', processors='tokenize,pos,lemma', use_gpu=True)
        self.tokenizer = AutoTokenizer.from_pretrained('EMBEDDIA/crosloengual-bert')
        self.model = AutoModelForSequenceClassification.from_pretrained('EMBEDDIA/crosloengual-bert',
                                                                       output_hidden_states=True)
        self.model.eval()


    def get_words_embeddings(self, word, sentences):
        word_lemma = get_word_lemmas_list([word], lemmatizer=self.lemmatizer)[0]
        sentences_lemmatized = get_word_lemmas_list(sentences[:], lemmatizer=self.lemmatizer)
        skipped_idx = []

        for i in range(len(sentences_lemmatized)):
            s = sentences_lemmatized[i]
            if word_lemma not in s:
                matches = get_close_matches(word_lemma, s.split(" "), n=1)
                if len(matches) == 0:
                    skipped_idx.append(i)
                else:
                    sentences_lemmatized[i] = s.replace(matches[0], word_lemma)

        indices = [s[:s.index(word_lemma) + 1].count(" ") for s in sentences_lemmatized if word_lemma in s]

        if len(indices) == 0:
            return []

        # TODO: delete skipped indices

        sentences, dataset, indices, labels = prepare_data(word, [], indices, sentences, self.tokenizer)

        with torch.no_grad():
            outputs = self.model(**dataset, output_hidden_states=True)

        results = get_embeddings_from_results(outputs, labels, word, sentences, indices, False)

        return [r[-1] for r in results], skipped_idx