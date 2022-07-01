import os

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from datasets import load_metric

import file_helpers
from classification.dataset import parse_embeddings_data, parse_sentence_data
from classification.scoring import b_metrics
from data.embeddings import get_token_range

#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)


class AntSynModel:

    def __init__(self, model_name="EMBEDDIA/crosloengual-bert", tokenizer_path=None):
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=2, output_attentions = False, output_hidden_states = False)
        self.model.eval()
        self.metric = load_metric('glue', 'sst2')
        #self.checkpoint = 'model/cse_full_model.ckpt'

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=labels)

    def split_data(self, data_file, ratio=0.2, batch_size=5):
        token_id, attention_masks, labels  = self.get_data(data_file, limit_range=(0, 10))

        train_idx, val_idx = train_test_split(
            np.arange(len(labels)),
            test_size=ratio,
            shuffle=True,
            stratify=labels)

        train_set = TensorDataset(token_id[train_idx],
                                  attention_masks[train_idx],
                                  labels[train_idx])

        val_set = TensorDataset(token_id[val_idx],
                                attention_masks[val_idx],
                                labels[val_idx])

        train_dataloader = DataLoader(
            train_set,
            sampler=RandomSampler(train_set),
            batch_size=batch_size
        )

        validation_dataloader = DataLoader(
            val_set,
            sampler=SequentialSampler(val_set),
            batch_size=batch_size
        )

        return train_dataloader, validation_dataloader

    def finetune(self, data_file, model_name, out_dir=None, epochs=2):
        if out_dir:
            file_path = os.path.join(out_dir, model_name)

        train_dataloader, val_dataloader = self.split_data(data_file)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5, eps=1e-08)

        print("finetuning ...")
        for i in range(epochs):
            print(f'epoch {i}')

            tr_loss, n_tr_examples, n_tr_steps = self.train(train_dataloader, save=True, file_path=file_path)
            val_accuracy, val_precision, val_recall, val_specificity = self.validate(train_dataloader)

            print('\n\t - Train loss: {:.4f}'.format(tr_loss / n_tr_steps))
            print('\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy) / len(val_accuracy)))
            print('\t - Validation Precision: {:.4f}'.format(sum(val_precision) / len(val_precision)) if len(
                val_precision) > 0 else '\t - Validation Precision: NaN')
            print('\t - Validation Recall: {:.4f}'.format(sum(val_recall) / len(val_recall)) if len(
                val_recall) > 0 else '\t - Validation Recall: NaN')
            print('\t - Validation Specificity: {:.4f}\n'.format(sum(val_specificity) / len(val_specificity)) if len(
                val_specificity) > 0 else '\t - Validation Specificity: NaN')

    def train(self, train_dataloader, save=True, file_path=None):
        self.model.train()

        tr_loss = 0
        n_tr_examples, n_tr_steps = 0, 0

        for step, batch in enumerate(train_dataloader):

            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            self.optimizer.zero_grad()

            # Forward pass
            train_output = self.model(b_input_ids,
                                      token_type_ids=None,
                                      attention_mask=b_input_mask,
                                      labels=b_labels)
            # Backward pass
            train_output.loss.backward()
            self.optimizer.step()

            # Update tracking variables
            tr_loss += train_output.loss.item()
            n_tr_examples += b_input_ids.size(0)
            n_tr_steps += 1

        if save:
            torch.save(self.model, file_path)

        return tr_loss, n_tr_examples, n_tr_steps

    def validate(self, validation_dataloader):
        self.model.eval()

        val_accuracy = []
        val_precision = []
        val_recall = []
        val_specificity = []

        for batch in validation_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                # Forward pass
                eval_output = self.model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask)

            logits = eval_output.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate validation metrics
            b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
            val_accuracy.append(b_accuracy)
            if b_precision != 'nan': val_precision.append(b_precision)
            if b_recall != 'nan': val_recall.append(b_recall)
            if b_specificity != 'nan': val_specificity.append(b_specificity)

        return val_accuracy, val_precision, val_recall, val_specificity

    def predict(self, data_file, batch_size=4, limit_range=None):
        data = self.get_data(data_file, limit_range=(10, 15)) # (n//5, n)
        y = to_categorical(data[2], num_classes=2)

        print('predicting')
        outputs = self.model(input_ids=data[0], attention_mask=data[1])

        # Extract the last hidden state of the token `[CLS]` for classification task
        probs = outputs[0].detach().numpy()
        print(probs)

        for i, p in zip(y, probs):
            print("y, probs:", i, p)

    def get_data(self, filename, limit_range=None):
        self.data_dict = parse_sentence_data(filename, limit_range)
        self.trim_sentences()
        self.add_special_tokens(mark_word=False)

        labels = torch.tensor(self.data_dict['labels'], dtype=torch.long)

        encodings = [self.tokenizer.encode_plus(" ".join(s), truncation=True, max_length=512, add_special_tokens=False,
                                                padding=True, pad_to_multiple_of=512) for s in self.data]
        print("result:", self.tokenizer.decode(encodings[0]['input_ids']))

        input_ids = torch.tensor([e.get('input_ids') for e in encodings])
        attention_masks = torch.tensor([e.get('attention_mask') for e in encodings])

        return input_ids, attention_masks, labels

    def add_special_tokens(self, mark_word=False):
        if mark_word:
            self.data = []
            for ss, ii in zip(self.data_dict['sentence_pairs'], self.data_dict['index_pairs']):
                s1, s2 = ss
                i1, i2 = ii

                s1_split, s2_split = s1.split(" "), s2.split(" ")
                s1_new = s1_split[:i1[0]] + ["[BOW]"] + s1_split[i1[0]: i1[1]] + ["[EOW]"] + s1_split[i1[1]:] + ["SEP"]
                s2_new = s2_split[:i2[0]] + ["[BOW]"] + s2_split[i2[0]: i2[1]] + ["[EOW]"] + s2_split[i2[1]:]

                self.data.append(["CLS"] + s1_new + s2_new)
        else:
            self.data = [f"[CLS] {s1} [SEP] {s2} [SEP]".split(" ") for s1, s2 in self.data_dict['sentence_pairs']]

    def trim_sentences(self):
        word_pairs = self.data_dict['word_pairs']
        sentence_pairs = self.data_dict['sentence_pairs']
        index_pairs = self.data_dict['index_pairs']

        indices_list = [x for i in index_pairs for x in i]
        sentences_list = [x for s in sentence_pairs for x in s]
        words_list = [x for s in word_pairs for x in s]
        n = len(sentences_list)

        for i in range(n):
            new_sentence, _, word_indices = get_token_range(sentences_list[i].split(" "), words_list[i], indices_list[i], self.tokenizer)
            sentences_list[i] = new_sentence
            indices_list[i] = word_indices

        self.data_dict['sentence_pairs'] = [(sentences_list[2*i], sentences_list[2*i + 1]) for i in range(n // 2)]
        self.data_dict['index_pairs'] =  [(indices_list[2*i], indices_list[2*i + 1]) for i in range(n // 2)]


    def update_tokenizer(self, new_tokens=["[BOW]", "[EOW]"], out_dir="model/saved"):
        self.tokenizer.add_tokens(new_tokens, special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.save_pretrained(out_dir)

