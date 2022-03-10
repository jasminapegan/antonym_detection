from typing import List

import classla

def lemmatize_source(source_file, lemmatized_file):
    classla.download('sl')

    with open(source_file, "r", encoding="utf8") as in_file:
        with open(lemmatized_file, "w", encoding="utf8") as outf:

            data = []
            for i, line in enumerate(in_file):

                word_data = line.split('|')

                # write header
                if i == 0:
                    outf.write("\t".join(["lemma"] + word_data))

                else:
                    word_data[0] = word_data[0].split(" ")[0]
                    word_data[0] = word_data[0].split("-")[0]
                    data.append([""] + word_data)

            for line in get_word_lemmas(data):
                outf.write("\t".join(line))


def get_word_lemmas(words_data):
    lemmatizer = classla.Pipeline('sl', processors='tokenize,pos,lemma', use_gpu=True)
    doc = lemmatizer(" ".join([word_data[1] for word_data in words_data]))

    for sentence in doc.sentences:
        for i, word in enumerate(sorted(sentence.words, key=lambda x: x.id)):

            words_data[i][0] = word.lemma
            if words_data[i][0] != words_data[i][1]:
                print(words_data[i])

    return words_data


def get_word_lemmas_list(words_list: List[str], lemmatizer: classla.Pipeline=None) -> List[str]:
    """
    Lemmatizes phrases in 'words_list' word by word. Caution: phrases containing '-' will loose it after lemmatization.

    :param words_list: a list of words and phrases to lemmatize
    :param lemmatizer: preexisting lemmatizer for efficiency
    :return: a list of lemmatized phrases
    """

    if not lemmatizer:
        lemmatizer = classla.Pipeline('sl', processors='tokenize,pos,lemma', use_gpu=True)

    lemmatized = []
    for word in words_list:
        doc = lemmatizer(word)
        lemmas = []

        for sentence in doc.sentences:
            lemma = []

            for i, word in enumerate(sorted(sentence.words, key=lambda x: x.id)):
                lemma.append(word.lemma)

            lemmas.append(" ".join(lemma))

        lemmatized.append(" ".join(lemmas))

    return lemmatized
