import tensorflow as tf
import numpy as np
import sys
import torch
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Activation, TimeDistributed, Masking, GRU
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from ast import literal_eval
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
from random import shuffle
from transformers import AutoTokenizer, AutoModel, BertModel
import h5py
from ast import literal_eval
import numpy as np
import sys
import json

# Best value: 0.9707831

# train_indices = range(10)
# test_indices = range(10)
# train_indices = range(28180)
# test_indices =  range(22289, 28180)

OUT_FILENAME = './dict_string.py'
SKIPPED_FILENAME = './skipped_file_parseme_pl.txt'
np.set_printoptions(threshold=sys.maxsize)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def get_elmo_vectors(FILENAMES):
    all_vectors = []
    for FILENAME in FILENAMES:
        with h5py.File(FILENAME, 'r') as f, open(OUT_FILENAME, 'w', encoding='utf-8') as outf, open(SKIPPED_FILENAME,
                                                                                                    'w',
                                                                                                    encoding='utf-8') as skipf:
            # List all groups
            print("Keys: %s" % f.keys())

            keys = list(f.keys())
            json_acceptable_string = str(f['sentence_to_index'][0]).replace("'", "\"").replace('\\\\', '\\').replace(
                '\\\\', '')[2:-1]
            sentence_to_index_map = json.loads(json_acceptable_string)
            index_to_sentence_map = {v: k for k, v in sentence_to_index_map.items()}

            keys_int = [int(x) for x in keys if x != 'sentence_to_index']
            test = [x for x in keys]

            print(keys_int)
            print('done 0')

            sorted_keys_str = [str(x) for x in sorted(keys_int)]
            print(sorted_keys_str)
            print('done 0')
            print(len(sentence_to_index_map.keys()), len(index_to_sentence_map.keys()), len(test), len(keys_int),
                  len(sorted_keys_str))
            print('done 1')
            num_skipped = 0
            for k in sorted_keys_str:
                vectors = list(f[k])
                all_vectors.append(vectors)
                if k not in index_to_sentence_map.keys():
                    num_skipped += 1
                    print('skipped', k, num_skipped, len(vectors))
                    # print(k, file=skipf)
                    continue
                words = index_to_sentence_map[k].split(' ')
            print('done skipped', num_skipped)
    print(all_vectors[0])
    print('Got', len(all_vectors), 'elmo vectors')
    return all_vectors


# VECTOR_DIM = 1024
VECTOR_DIM = 768  # bert
MAX_SEQUENCE_LEN = 50

IN_FILENAME_TEST = None
NUM_CLASSES = 4


def expand_classes_for_bert(sent, sent_classes):
    new_sent_classes = []
    curr_cls_index = -1
    for i in range(len(sent)):
        if sent[i][0] != '#':
            curr_cls_index += 1
            new_sent_classes.append(sent_classes[curr_cls_index])

        else:
            new_sent_classes.append(sent_classes[curr_cls_index])
    return new_sent_classes


def get_xy_all(filenames, tokenizer, model):
    num_prints = 0
    with open('./sentences_new_attempt.txt', 'w', encoding='utf-8') as outf:
        data_by_expressions = {}
        all_data = []
        sent_wide_Y = []
        sents_X = []
        sents_Y = []
        curr_sent_X = []
        curr_sent_Y = []
        all_sents = []
        expressions = []
        curr_sent_words = []
        print('starting')
        CLS_TO_INT_DICT = {'NE': 1, 'DA': 2, '*': 3, 'NEJASEN_ZGLED': 4}
        classes = []
        words = []
        X = []
        Y = []
        for filename in filenames:
            print('reading file', filename)
            with open(filename, 'r', encoding='utf-8') as f:
                debug_sent = []
                for i, line in enumerate(f):
                    if i % 5000 == 0:
                        print(i)
                    # if i >= 1500:
                    #    break
                    parts = line.split('\t')
                    word = parts[0]
                    # print(len(parts))
                    if len(word) == 0:
                        continue
                    if len(parts) != 3:
                        continue

                    # print('len of parts', len(parts))
                    word = parts[0]
                    cls = parts[1]
                    expression = parts[2]
                    debug_sent.append((word, cls, expression))
                    # print(word, cls)
                    classes.append(cls)
                    words.append(word)
                    # print(exp, vector[:10], cls, expression)
                    if not (cls == 'DA' or cls == 'NE' or cls == '*'):
                        continue
                    curr_sent_words.append(word)
                    # print(len(literal_eval(vector)))
                    curr_sent_Y.append(CLS_TO_INT_DICT[cls])
                    if word[-1] == '.':
                        # print('curr sent words', curr_sent_words)
                        str_sentence = ' '.join([x for x in curr_sent_words])
                        basic_tokens = [x for x in curr_sent_words]
                        tokenized_text = tokenizer.tokenize(str_sentence)
                        # tokenized_text = basic_tokens
                        tokenized_text = [x for x in tokenized_text if
                                          x not in [',', ':', ';', '!', '?', '.', '"', "'", '/', '\\']]
                        if num_prints == 0:
                            print(tokenized_text)
                            num_prints += 1
                        # if len(basic_tokens) == len(tokenized_text):
                        # print('tokenizer didn\'t do anything')
                        # print(basic_tokens)
                        # print(tokenized_text)
                        if len(tokenized_text) > 510:
                            print('skipped one')
                            tokenized_text = tokenized_text[:510]
                            # curr_sent_words = []
                            # curr_sent_X = []
                            # curr_sent_Y = []
                            # debug_sent = []
                            # continue
                        # print(expression)
                        expanded_classes = []
                        current_class_index = -1
                        # print(basic_tokens)
                        for w in tokenized_text:
                            # print(w[0:2])
                            if w[0:2] == '##':
                                expanded_classes.append(curr_sent_Y[current_class_index])
                                # print(w, curr_sent_Y[current_class_index], current_class_index, len(curr_sent_Y))
                            else:
                                current_class_index += 1
                                expanded_classes.append(curr_sent_Y[current_class_index])
                                # print(w, curr_sent_Y[current_class_index], current_class_index, len(curr_sent_Y))

                        # for x, y in zip(tokenized_text, expanded_classes):
                        #    print(x,y)
                        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                        tokens_tensor = torch.tensor([indexed_tokens])
                        with torch.no_grad():
                            outputs = model(tokens_tensor)
                            predictions = outputs[0]
                        vectors = predictions.numpy()[0]
                        if CLS_TO_INT_DICT['DA'] in curr_sent_Y:
                            sent_wide_cls = CLS_TO_INT_DICT['DA']
                        elif CLS_TO_INT_DICT['NE'] in curr_sent_Y:
                            sent_wide_cls = CLS_TO_INT_DICT['NE']
                        else:
                            sent_wide_cls = CLS_TO_INT_DICT['NEJASEN_ZGLED']
                            # print('debug sent', debug_sent)

                        if sent_wide_cls == CLS_TO_INT_DICT['NE'] or sent_wide_cls == CLS_TO_INT_DICT['DA']:
                            all_data.append((vectors, sent_wide_cls))
                            # all_data.append((vectors, expand_classes_for_bert(tokenized_text, curr_sent_Y)))
                            all_sents.append(tokenized_text)
                        sent_wide_cls = None
                        # print(str_sentence, file=outf)
                        sent_wide_Y.append(sent_wide_cls)
                        debug_sent = []
                        curr_sent_X = []
                        curr_sent_words = []
                        curr_sent_Y = []
                    # if cls == '5':
                    #    print(line)
                X = np.array(X)
                Y = np.array(Y)
                sents_X = np.array(sents_X)
                sents_Y = np.array(sents_Y)

            print(Counter(classes))
            # print(Counter(words))
            # print(X.shape)
            # print(Y.shape)
            # print(sents_X.shape)
            # print(sents_Y.shape)
        # return sents_X, sents_Y
    return all_data, all_sents


def get_xy_all_elmo_vectors(filenames, elmo_vectors):
    curr_elmo_i = 0
    with open('./sentences_new_attempt.txt', 'w', encoding='utf-8') as outf:
        data_by_expressions = {}
        all_data = []
        sent_wide_Y = []
        sents_X = []
        all_sents = []
        sents_Y = []
        curr_sent_X = []
        curr_sent_Y = []
        expressions = []
        curr_sent_words = []
        print('starting')
        CLS_TO_INT_DICT = {'NE': 1, 'DA': 2, '*': 3, 'NEJASEN_ZGLED': 4}
        classes = []
        words = []
        X = []
        Y = []
        for filename in filenames:
            print('reading file', filename)
            with open(filename, 'r', encoding='utf-8') as f:
                debug_sent = []
                for i, line in enumerate(f):
                    if i % 5000 == 0:
                        print(i)
                    # if i >= 1500:
                    #    break
                    parts = line.split('\t')
                    word = parts[0]
                    # print(len(parts))
                    if len(word) == 0:
                        continue
                    if len(parts) != 3:
                        continue

                    # print('len of parts', len(parts))
                    word = parts[0]
                    cls = parts[1]
                    expression = parts[2]
                    debug_sent.append((word, cls, expression))
                    # print(word, cls)
                    classes.append(cls)
                    words.append(word)
                    # print(exp, vector[:10], cls, expression)
                    if not (cls == 'DA' or cls == 'NE' or cls == '*'):
                        continue
                    curr_sent_words.append(word)
                    # print(len(literal_eval(vector)))
                    curr_sent_Y.append(CLS_TO_INT_DICT[cls])
                    if word[-1] == '.':
                        # print('curr sent words', curr_sent_words)
                        str_sentence = ' '.join([x for x in curr_sent_words])
                        basic_tokens = [x for x in curr_sent_words]
                        tokenized_text = basic_tokens
                        tokenized_text = [x for x in tokenized_text if
                                          x not in [',', ':', ';', '!', '?', '.', '"', "'", '/', '\\']]
                        # if len(basic_tokens) == len(tokenized_text):
                        # print('tokenizer didn\'t do anything')
                        # print(basic_tokens)
                        # print(tokenized_text)
                        if len(tokenized_text) > 510:
                            print('skipped one')
                            tokenized_text = tokenized_text[:510]
                        #    curr_sent = []
                        #    curr_sent_words = []
                        #    curr_sent_X = []
                        #    curr_sent_Y = []
                        #    debug_sent = []
                        #    continue
                        # print(expression)
                        expanded_classes = []
                        current_class_index = -1
                        # print(basic_tokens)
                        for w in tokenized_text:
                            # print(w[0:2])
                            if w[0:2] == '##':
                                expanded_classes.append(curr_sent_Y[current_class_index])
                                # print(w, curr_sent_Y[current_class_index], current_class_index, len(curr_sent_Y))
                            else:
                                current_class_index += 1
                                expanded_classes.append(curr_sent_Y[current_class_index])
                                # print(w, curr_sent_Y[current_class_index], current_class_index, len(curr_sent_Y))

                        # for x, y in zip(tokenized_text, expanded_classes):
                        #    print(x,y)
                        if CLS_TO_INT_DICT['DA'] in curr_sent_Y:
                            sent_wide_cls = CLS_TO_INT_DICT['DA']
                        elif CLS_TO_INT_DICT['NE'] in curr_sent_Y:
                            sent_wide_cls = CLS_TO_INT_DICT['NE']
                        else:
                            sent_wide_cls = CLS_TO_INT_DICT['NEJASEN_ZGLED']
                            # print('debug sent', debug_sent)

                        if sent_wide_cls == CLS_TO_INT_DICT['NE'] or sent_wide_cls == CLS_TO_INT_DICT['DA']:
                            all_data.append((elmo_vectors[curr_elmo_i], sent_wide_cls))
                            # all_data.append((elmo_vectors[curr_elmo_i], curr_sent_Y))
                            all_sents.append(tokenized_text)
                        sent_wide_cls = None
                        print(str_sentence, file=outf)
                        sent_wide_Y.append(sent_wide_cls)
                        debug_sent = []
                        curr_sent_X = []
                        curr_sent_words = []
                        curr_sent_Y = []
                        curr_elmo_i += 1
                    # if cls == '5':
                    #    print(line)
                X = np.array(X)
                Y = np.array(Y)
                sents_X = np.array(sents_X)
                sents_Y = np.array(sents_Y)

            print(Counter(classes))
            # print(Counter(words))
            # print(X.shape)
            # print(Y.shape)
            # print(sents_X.shape)
            # print(sents_Y.shape)
        # return sents_X, sents_Y
    return all_data, all_sents


def bert_tensorflow_test(X_train, X_test, Y_train, Y_test, train_indices, test_indices, out_filename, cp_callback,
                         checkpoint_filename=None):
    # Model
    model = Sequential()

    # model.add(Masking(mask_value=0.0, input_shape=(MAX_SEQUENCE_LEN,VECTOR_DIM)))
    model.add(Masking(mask_value=0.0, dtype='float64'))
    # forward_layer = LSTM(200, return_sequences=True)
    forward_layer = GRU(10, return_sequences=False, dropout=0.5)
    # backward_layer = LSTM(200, activation='relu', return_sequences=True,
    backward_layer = GRU(10, return_sequences=False, dropout=0.5,
                         go_backwards=True)
    model.add(Bidirectional(forward_layer, backward_layer=backward_layer))  # ,
    # input_shape=(MAX_SEQUENCE_LEN,VECTOR_DIM)))
    # model.add(TimeDistributed(Dense(NUM_CLASSES)))
    # Remove TimeDistributed() so that predictions are now made for the entire sentence
    # model.add(TimeDistributed(Dense(NUM_CLASSES)))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))
    # print('preds shape', model.predict(X_train[:3]).shape)
    # print('Y_train shape', Y_train[:3].shape)
    # print(list(Y_train[:3]))
    classes = []
    for y in Y_train:
        cls = np.argmax(y)
        classes.append(cls)
    print(Counter(classes))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print('compiled model')

    X_train = np.array(X_train).astype('float32')
    Y_train = np.array(Y_train).astype('float32')
    print(Y_train[0])
    print(X_train.shape)
    print(Y_train.shape)
    if checkpoint_filename == None:
        # model.fit(X_train, Y_train, batch_size=4, epochs=3, validation_split=0.0, callbacks=[cp_callback])
        print('fit model')
    else:
        print('loading model')
        model.load_weights(checkpoint_filename)
        print('loaded model from', checkpoint_filename)
    print(len(X_test), len(Y_test))
    eval = model.evaluate(np.array(X_test), np.array(Y_test), batch_size=4)
    # print('X_test[0]')
    # print(X_test[0])
    # print(X_train[0])
    print('predicting')
    preds = model.predict_proba(np.array(X_test), verbose=1, batch_size=4)
    for i, p in zip(test_indices, preds):
        print(i, p)
    preds_train = model.predict_proba(np.array(X_train), verbose=1, batch_size=4)

    # raise TypeError
    # print(preds)
    num_correct = 0
    num_incorrect = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    # idiomatic = 2, non-idiomatic = 3
    correct = 0
    incorrect = 0
    with open(out_filename + '.txt', 'w') as outf:
        print(preds, file=outf)
    with open(out_filename + '_y_test.txt', 'w') as outf:
        print(Y_test, file=outf)
    with open(out_filename + '_y_train.txt', 'w') as outf:
        print(Y_test, file=outf)
    with open('preds_out_temp.txt', 'w') as tempoutf:
        for pred, y in zip(preds, Y_test):
            for pp, yy in zip(pred, y):
                print(pp, yy, np.argmax(pp), np.argmax(yy))
                if np.argmax(yy) != 0 and np.argmax(pp) != 0:
                    if np.argmax(pp) == np.argmax(yy):
                        correct += 1
                    else:
                        incorrect += 1
                    if np.argmax(pp) == 2 and np.argmax(yy) == 2:
                        TP += 1
                    if np.argmax(pp) != 2 and np.argmax(yy) != 2:
                        TN += 1
                    if np.argmax(pp) == 2 and np.argmax(yy) != 2:
                        FP += 1
                    if np.argmax(pp) != 2 and np.argmax(yy) == 2:
                        FN += 1
    with open(out_filename[:-4] + '_f1scores.txt', 'w') as f:

        f1 = 0
        print('TP', TP, file=f)
        print('TN', TN, file=f)
        print('FP', FP, file=f)
        print('FN', FN, file=f)
        if TP == 0:
            precision = 0
            recall = 0
            print('precision', 0, file=f)
            print('recall', 0, file=f)
            print('F1 score', 0, file=f)
            f1 = 0
            custom_accuracy = 0
        else:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            print('precision', TP / (TP + FP), file=f)
            print('recall', TP / (TP + FN), file=f)
            print('F1 score', (2 * precision * recall) / (precision + recall), file=f)
            f1 = (2 * precision * recall) / (precision + recall)
            custom_accuracy = correct / (correct + incorrect)
        print('custom accuracy is', custom_accuracy, file=f)
        print('eval is', eval, file=f)
        # raise TypeError
        for y in Y_test:
            cls = np.argmax(y)
            classes.append(cls)
        class_nums = Counter(classes)
        print(class_nums)
        default_acc = class_nums[1] / (class_nums[1] + class_nums[2])
        print('default accuracy is', default_acc, 'or', 1 - default_acc)
        print('eval is', eval)
        print('f1 is', f1)
    return eval, custom_accuracy, default_acc, custom_accuracy, preds, preds_train


def get_already_processed(filename):
    if filename == None:
        return set([])
    already_processed = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.split(' ')
            if words[0] == 'EXP':
                already_processed.append(' '.join(words[1:]))
    return set(already_processed)


# ELMO_VECTORS_FILENAME = ['./elmo_vectors_new_attempt.txt']
IN_FILENAMES = ['./classes_elmo_1_small.txt']
# IN_FILENAMES = ['./classes_elmo_1.txt',
#                './classes_elmo_2.txt',
#                './classes_elmo_3.txt']


# ZA ELMO
# elmo_vectors = get_elmo_vectors(ELMO_VECTORS_FILENAME)
# elmo_ys = get_elmo_ys(IN_FILENAMES)
# data_elmo, sents_elmo = get_xy_all_elmo_vectors(IN_FILENAMES, elmo_vectors)
# indices = list(range(len(data_elmo)))

# ZA CSE
model_crosloengual = AutoModel.from_pretrained("EMBEDDIA/crosloengual-bert")
tokenizer_crosloengual = AutoTokenizer.from_pretrained("EMBEDDIA/crosloengual-bert")
data_crosloengual, sents_crosloengual = get_xy_all(IN_FILENAMES, tokenizer_crosloengual, model_crosloengual)
model_checkpoint = 'cse_full_model.ckpt'
indices = list(range(len(data_crosloengual)))

# ZA MBERT
# data_mbert, sents_mbert = get_xy_all(IN_FILENAMES, tokenizer_mbert, model_mbert)


train_indices = indices[1:10]
test_indices = indices

for all_data, name, chkpt in zip([data_crosloengual], ['cse_check'], [None]):
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./' + name + '.ckpt',
                                                     save_weights_only=True,
                                                     verbose=1)

    f1s = []
    accs = []
    # exit()

    all_X = [x[0] for x in all_data]
    all_Y = [x[1] for x in all_data]
    all_Y = to_categorical(all_Y, num_classes=NUM_CLASSES)
    all_X = pad_sequences(all_X, padding='post', truncating='post', maxlen=MAX_SEQUENCE_LEN, dtype='float', value=0.0)
    all_train_X = [all_X[i] for i in train_indices]
    all_test_X = [all_X[i] for i in test_indices]
    all_train_Y = [all_Y[i] for i in train_indices]
    all_test_Y = [all_Y[i] for i in test_indices]

    results = bert_tensorflow_test(all_train_X, all_test_X, all_train_Y, all_test_Y, train_indices, test_indices,
                                   './in_test_' + name, cp_callback, checkpoint_filename=model_checkpoint)

    custom_accuracy = results[3]
    preds = results[4]
    preds_train = results[5]

    print('first 10 predictions')
    print(preds[:10])
    print('eval is', results[0])
    print('default accuracy is', results[2], 'or', 1 - results[2])
    print('num train is', len(all_train_X))
    print('num test is', len(all_test_X))
