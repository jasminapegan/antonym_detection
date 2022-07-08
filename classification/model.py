import os

import torch
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from datasets import load_metric
from matplotlib import pyplot as plt

from classification.dataset import parse_sentence_data
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


    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=labels)

    def prepare_data(self, data_file, ratio=0.2, batch_size=5):
        token_id, attention_masks, labels = self.get_data(data_file)

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

    def prepare_data_crossval(self, train_fname, val_fname, batch_size=5, n=3, mark_word=False):
        train_dataloaders = []
        val_dataloaders = []

        for i in range(n):
            train_file, val_file = f"{train_fname}{i}.txt", f"{val_fname}{i}.txt"

            token_id, attention_masks, labels  = self.get_data(train_file, mark_word=mark_word)
            train_set = TensorDataset(token_id, attention_masks, labels)
            train_dataloader = DataLoader(
                train_set,
                sampler=RandomSampler(train_set),
                batch_size=batch_size
            )
            train_dataloaders.append(train_dataloader)

            token_id, attention_masks, labels  = self.get_data(val_file)
            val_set = TensorDataset(token_id, attention_masks, labels)
            validation_dataloader = DataLoader(
                val_set,
                sampler=SequentialSampler(val_set),
                batch_size=batch_size
            )
            val_dataloaders.append(validation_dataloader)

        return train_dataloaders, val_dataloaders

    def finetune(self, val_dataloader, train_dataloader, outf, epochs=1, lr=2e-5, out_path=None):
        outf.write(f"Model with learning rate {lr}")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, eps=1e-08)

        self.tr_loss = []
        self.val_acc, self.val_prec, self.val_rec, self.val_spec = [], [], [], []
        self.n_tr_steps = []
        self.lr = lr

        for i in range(epochs):
            print(f'epoch {i}')
            outf.write(f'Epoch {i}\n')

            self.name = f"{lr}_{i}"
            model_path = os.path.join(out_path, self.name)

            # training metrics
            tr_loss, n_tr_examples, n_tr_steps = self.train(train_dataloader, out_model=model_path)
            self.tr_loss.append(tr_loss / n_tr_steps)
            outf.write('\n\t - Train loss: {:.4f}\n'.format(tr_loss / n_tr_steps))

            # validation metrics
            val_accuracy, val_precision, val_recall, val_specificity = self.validate(val_dataloader)
            avg_acc = sum(val_accuracy) / len(val_accuracy) if len(val_accuracy) > 0 else 0
            avg_prec = sum(val_precision) / len(val_precision) if len(val_precision) > 0 else 0
            avg_rec = sum(val_recall) / len(val_recall) if len(val_recall) > 0 else 0
            avg_spec = sum(val_specificity) / len(val_specificity) if len(val_specificity) > 0 else 0

            self.val_acc.append(avg_acc)
            self.val_prec.append(avg_prec)
            self.val_rec.append(avg_rec)
            self.val_spec.append(avg_spec)

            outf.write('\t - Validation Accuracy: {:.4f}\n'.format(avg_acc))
            outf.write('\t - Validation Precision: {:.4f}\n'.format(avg_prec))
            outf.write('\t - Validation Recall: {:.4f}\n'.format(avg_rec))
            outf.write('\t - Validation Specificity: {:.4f}\n'.format(avg_spec))

        plot_scores(self, out_path)

    def train(self, train_dataloader, out_model=None):
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

            #tr_acc += train_output.logits.argmax().item()

        #self.train_score = b_metrics(preds, b_labels)

        if out_model:
            torch.save(self.model, out_model)

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
            val_precision.append(b_precision)
            val_recall.append(b_recall)
            val_specificity.append(b_specificity)

        return val_accuracy, val_precision, val_recall, val_specificity

    def get_data(self, filename, limit_range=None, shuffle_data=True, mark_word=False):
        self.data_dict = parse_sentence_data(filename, limit_range, shuffle_lines=shuffle_data)
        self.trim_sentences()
        self.add_special_tokens(mark_word=mark_word)

        labels = torch.tensor(self.data_dict['labels'], dtype=torch.long)

        encodings = [self.tokenizer.encode_plus(" ".join(s), truncation=True, max_length=512, add_special_tokens=True,
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
                #s1_new = s1_split[:i1[0]] + ["[BOW]"] + s1_split[i1[0]: i1[1] + 1] + ["[EOW]"] + s1_split[i1[1] + 1:] + ["[SEP]"]
                #s2_new = s2_split[:i2[0]] + ["[BOW]"] + s2_split[i2[0]: i2[1] + 1] + ["[EOW]"] + s2_split[i2[1] + 1:]

                #self.data.append(["[CLS]"] + s1_new + s2_new)

                s1_new = s1_split[:i1[0]] + ["[BOW]"] + s1_split[i1[0]: i1[1] + 1] + ["[EOW]"] + s1_split[i1[1] + 1:] + ["[SEP]"]
                s2_new = s2_split[:i2[0]] + ["[BOW]"] + s2_split[i2[0]: i2[1] + 1] + ["[EOW]"] + s2_split[i2[1] + 1:]

                self.data.append(s1_new + s2_new)
        else:
            #self.data = [f"[CLS] {s1} [SEP] {s2} [SEP]".split(" ") for s1, s2 in self.data_dict['sentence_pairs']]
            self.data = [f"{s1} [SEP] {s2}".split(" ") for s1, s2 in self.data_dict['sentence_pairs']]

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
        self.tokenizer.add_tokens(new_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.save_pretrained(out_dir)

    def find_best(self, train_filename, val_filename, out_path, lrs=[5e-5, 3e-5, 2e-5], epochs=3, mark_word=False):
        train_dataloaders, val_dataloaders = self.prepare_data_crossval(train_filename, val_filename, mark_word=mark_word)

        max_accuracy = -1
        idx = -1
        best_model = None

        with open(os.path.join(out_path, "info.txt"), "w", encoding="utf8") as f:
            f.write(out_path + "\n")

            for lr in lrs:
                avg_val_acc = run_crossval_models(train_dataloaders, val_dataloaders, out_path, f, lr, epochs)

                if max(avg_val_acc) > max_accuracy:
                    idx = np.argmax(avg_val_acc)
                    max_accuracy = avg_val_acc[idx]
                    best_model = f"{lr}_{idx}"

        print(f"Best model {best_model} at epoch {idx} with val_accuracy: {max_accuracy}")

def run_crossval_models(train_dataloaders, val_dataloaders, out_path, f, lr, n_epochs):
    models = []
    for train_dataloader, val_dataloader in zip(train_dataloaders, val_dataloaders):
        model = run_model(val_dataloader, train_dataloader, out_path, f, lr, n_epochs)
        models.append(model)

    val_acc = [m.val_acc for m in models]
    val_acc_by_epoch = [[x[i] for i in range(len(val_acc[0]))] for x in val_acc]
    avg_val_acc = [sum(x) / len(x) for x in val_acc_by_epoch]

    return avg_val_acc

def run_model(val_dataloader, train_dataloader, out_path, f, lr, n_epochs):
    model = AntSynModel()
    model.finetune(val_dataloader, train_dataloader, f, lr=lr, epochs=n_epochs, out_path=out_path)
    return model

def plot_scores(model, out_path):
    tr_loss = model.tr_loss
    val_acc, val_prec, val_rec, val_spec = model.val_acc, model.val_prec, model.val_rec, model.val_spec

    epochs = range(1, len(val_acc) + 1)
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    plt.subplot(2, 1, 1)
    plt.plot(epochs, tr_loss, 'r', label='Training loss')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Scores')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, val_prec, 'r', label='Validation precision')
    plt.plot(epochs, val_spec, 'b', label='Validation specificity')
    plt.plot(epochs, val_rec, 'g', label='Validation recall')
    plt.plot(epochs, val_acc, color='magenta', label='Validation accuracy')
    plt.title('Validation scores')
    plt.xlabel('Epochs')
    plt.legend(loc='lower right')

    fname = os.path.join(out_path, model.name + ".png")
    plt.savefig(fname)

