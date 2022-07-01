import itertools
from collections import Counter

import classla
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Dense, Activation, Masking, GRU
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.sequence import pad_sequences
from transformers import AutoModel, AutoTokenizer
import numpy as np

import file_helpers
from clustering.scoring import parse_cluster_file, parse_score_data
from data.embeddings import WordEmbeddings

#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)

class AntSynModel:

    def __init__(self):
        self.model = AutoModel.from_pretrained("EMBEDDIA/crosloengual-bert")
        self.tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/crosloengual-bert")
        self.checkpoint = 'model/cse_full_model.ckpt'

    def build_model_old(self, checkpoint_filename):
        # input_shape=(3, 8, 768)
        self.model = Sequential()
        self.model.add(Masking(mask_value=0.0, dtype='float64'))
        forward_layer = GRU(10, return_sequences=False, dropout=0.5)
        backward_layer = GRU(10, return_sequences=False, dropout=0.5,
                             go_backwards=True)
        self.model.add(Bidirectional(forward_layer, backward_layer=backward_layer))
        #self.model.add(Dense(4))
        self.model.add(Dense(3))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        print('compiled model')

        #self.model.build(input_shape=input_shape)
        #print(self.model.summary())
        #print('built model')

        #model.load_weights(checkpoint_filename, by_name=True, skip_mismatch=True).expect_partial()
        self.model.load_weights(checkpoint_filename).expect_partial()
        print('loaded model from', checkpoint_filename)

    def fit_old(self, data_file, batch_size=4, range=None):
        X, y = self.get_data(data_file, range=range)
        print(X.shape)
        print(X[0])
        y = to_categorical(y, num_classes=3)

        print('predicting')
        preds = self.model.predict(X, verbose=1, batch_size=batch_size)
        for i, p in zip(y, preds):
            print(i, p)

    def predict_old(self, data_file, batch_size=4, range=None):
        X, y = self.get_data(data_file, range=range)
        y = to_categorical(y, num_classes=3)

        print('predicting')
        preds = self.model.predict(X, verbose=1, batch_size=batch_size)
        for i, p in zip(y, preds):
            print(i, p)

    def get_data(self, filename, use_labels=True, range=None):
        data = []
        labels = []

        with open(filename, "r", encoding="utf8") as f:
            for i, line in enumerate(f):
                if not range or i in range:

                    data_line = parse_line(line, embeddings=True, only_embeddings=True, labels=use_labels)
                    if not data_line:
                        continue
                    data.append(data_line[0] + data_line[1])

                    if use_labels:
                        labels.append(data_line[2])

                if range and i > range[1]:
                    break

        data = np.array(data).astype('float32')
        data = data.reshape(3, data.shape[1] // 768, 768)
        print("data shape: ", data.shape)

        if use_labels:
            labels = np.array(labels).astype('int')
            return data, labels
        else:
            return data

def create_dataset(cluster_file, score_file_ant, score_file_syn, examples_file, out_file):
    clusters = parse_cluster_file(cluster_file)
    score_data_ant = parse_score_data(score_file_ant)
    score_data_syn = parse_score_data(score_file_syn)
    score_data = {x: ('ant', y) for x, y in score_data_ant.items()}
    score_data.update({x: ('syn', y) for x, y in score_data_syn.items()})
    examples_data = file_helpers.load_validation_file_grouped(examples_file, indices=True, embeddings=False, use_pos=False)

    with open(out_file, "w", encoding="utf8") as f:

        for pair_data in clusters.keys():

            cluster = clusters[pair_data]
            ant_syn, is_correct = score_data[pair_data]

            if "DA" in is_correct:
                #w1, w2, pos = pair_data
                w1, w2 = pair_data
                sense_w1, sense_w2 = cluster["w1_sense"], cluster["w2_sense"]
                w1_data, w2_data = examples_data[w1]["all"], examples_data[w2]["all"] #[pos]
                is_ant = 0 if ant_syn == "ant" else 1

                w1_examples = [(s, i) for l, s, i in zip(w1_data['labels'], w1_data['sentences'], w1_data['indices']) if l == sense_w1]
                w2_examples = [(s, i) for l, s, i in zip(w2_data['labels'], w2_data['sentences'], w2_data['indices']) if l == sense_w2]

                # w1, pos1, form1, l1, idx1, s1, w2, pos2, form2, l2, idx2, s2, (label)
                for s_i_1, s_i_2 in itertools.product(w1_examples, w2_examples):
                    s1, i1 = s_i_1
                    s2, i2= s_i_2
                    f1, f2 = s1.split(" ")[int(i1)], s2.split(" ")[int(i2)]
                    f.write(f"{w1}\t/\t{f1}\t{sense_w1}\t{i1}\t{s1}\t")
                    f.write(f"{w2}\t/\t{f2}\t{sense_w2}\t{i2}\t{s2}\t{is_ant}\n")

def get_bert_embeddings(sentence_file, embeddings_out_file):
    # w1, pos1, form1, label1, idx1, sentence1, w2, pos2, form2, label2, idx2, sentence2
    we = WordEmbeddings()
    lemmatizer =classla.Pipeline('sl', processors='tokenize,pos,lemma', use_gpu=True)

    with open(sentence_file, "r", encoding="utf8") as f:
        data = f.readlines()

    with open(embeddings_out_file, "w", encoding="utf8") as outf:
        for line in data:
            w1, pos1, form1, l1, idx1, s1, w2, pos2, form2, l2, idx2, s2, label = parse_line(line, embeddings=False)
            res1, res2 = we.get_words_embeddings([w1, w2], [pos1, pos2], [idx1, idx2], [s1, s2])
            e1 = " ".join([str(float(x)) for x in res1[-1]])
            e2 = " ".join([str(float(x)) for x in res2[-1]])
            outf.write(f"{w1}\t{pos1}\t{form1}\t{idx1}\t{l1}\t{s1}\t{e1}\t")
            outf.write(f"{w2}\t{pos2}\t{form2}\t{idx2}\t{l2}\t{s2}\t{e2}\t")
            outf.write(f"{label}\n")

def parse_line(line, embeddings=False, only_embeddings=False, labels=True):
    # w1, pos1, form1, l1, idx1, s1, e1, w2, pos2, form2, l2, idx2, s2, e2, (label)
    data = line.strip().split("\t")

    if not embeddings:
        # w1, pos1, form1, l1, idx1, s1, w2, pos2, form2, l2, idx2, s2, (label)
        assert 12 <= len(data) <= 13
        data[4], data[10] = int(data[4]), int(data[10])
        return data
    else:
        # w1, pos1, form1, l1, idx1, s1, e1, w2, pos2, form2, l2, idx2, s2, e2, (label)
        data[4], data[11] = int(data[4]), int(data[11])
        e1 = [float(x) for x in data[6].split(" ")]
        e2 = [float(x) for x in data[13].split(" ")]

        if labels:
            if len(data) != 15:
                return []
            if only_embeddings:
                return e1, e2, data[-1]
            else:
                return data
        else:
            assert len(data) == 14

            if only_embeddings:
                return e1, e2
            else:
                return data