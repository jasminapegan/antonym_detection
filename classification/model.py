import os

from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from datetime import timedelta

import torch
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from datasets import load_metric

from classification.dataset import parse_sentence_data
from classification.utils import compute_metrics, plot_scores
from data.embeddings import get_token_range

import file_helpers

#gpus = tf.config.experimental.list_physical_devices('GPU')
#print(gpus)
#tf.config.experimental.set_memory_growth(gpus[0], True)

#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#gpus = tf.config.experimental.list_physical_devices('GPU')
#gpus = tf.config.list_physical_devices('GPU')


class AntSynModel:

    def __init__(self, model_name="EMBEDDIA/crosloengual-bert", use_tokenizer=None, resize_model=False):
        if use_tokenizer:
            self.tokenizer = use_tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=2, output_attentions = False, output_hidden_states = False)

        if resize_model:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model_name = model_name
        modules = [self.model.base_model.embeddings, *self.model.base_model.encoder.layer[:7]] # up to 9 layers can be frozen
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

        self.model.eval()
        self.metric = load_metric('glue', 'sst2')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=labels)

    def prepare_data(self, tokenizer, data_file, ratio=0.2, batch_size=32):
        token_id, attention_masks, labels = get_data(data_file, tokenizer)

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

    def finetune(self, val_dataloader, train_dataloader, outf, epochs=1, lr=2e-5, out_path=None):
        outf.write(f"Model with learning rate {lr}\n")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, eps=1e-08)

        self.tr_loss = []
        self.val_f1_all = []
        self.n_tr_steps = []
        self.lr = lr

        for i in range(1, epochs + 1):
            print(f'[{file_helpers.get_now_string()}] Starting epoch {i}')
            outf.write(f'\nEpoch {i}\n')

            self.name = f"{lr}_{i}"
            model_path = os.path.join(out_path, self.name)

            # training metrics
            tr_loss, n_tr_examples, n_tr_steps = self.train(train_dataloader, out_model_path=model_path)
            self.tr_loss.append(tr_loss / n_tr_steps)
            outf.write('\n\t - Train loss: {:.4f}\n'.format(tr_loss / n_tr_steps))

            # validation metrics
            val_f1_scores = self.validate(val_dataloader)
            avg_f1 = sum(val_f1_scores) / len(val_f1_scores) if len(val_f1_scores) > 0 else 0
            self.val_acc.append(avg_f1)
            outf.write('\t - Validation Accuracy: {:.4f}\n'.format(avg_f1))

        plot_scores(self, out_path)

    def finetune_2(self, val_dataloader, train_dataloader, outf, epochs=1, lr=2e-5, out_path=None, out_name=None):

        # Total number of training steps
        total_steps = len(train_dataloader) * epochs

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, eps=1e-08)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        self.tr_loss_epoch = []
        self.val_f1_epoch = []
        self.n_tr_steps = []
        self.lr = lr

        trigger_f1 = 0
        max_f1 = 0.0

        self.metrics_by_epoch = {"loss": [],
                                 "val_acc": [],
                                 "val_f1": []}

        for i in range(1, epochs + 1):

            out_epoch_name = f"{out_name}_{i}"

            # Measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # Reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_counts = 0, 0, 0

            outf.write(f'Epoch {i}\n')

            # train
            self.model.train()

            tr_loss = 0
            n_tr_examples, n_tr_steps = 0, 0

            for step, batch in enumerate(train_dataloader):
                batch_counts += 1

                # Load batch to GPU
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                self.optimizer.zero_grad()

                # Forward pass
                train_output = self.model(b_input_ids,
                                          token_type_ids=None,
                                          attention_mask=b_input_mask,
                                          labels=b_labels)

                # Compute loss and accumulate the loss values
                loss = train_output.loss
                batch_loss += loss.item()
                total_loss += loss.item()

                # Backward pass
                loss.backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and the learning rate
                self.optimizer.step()
                self.scheduler.step()

                # Print the loss values and time elapsed
                if (step % 500 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                    # Calculate time elapsed
                    elapsed_sec = time.time() - t0_batch
                    time_elapsed = str(timedelta(seconds=elapsed_sec))
                    print(f"Epoch progress {step / len(train_dataloader)} Time elapsed: {time_elapsed}")

                    # Reset batch tracking variables
                    batch_loss, batch_counts = 0, 0
                    t0_batch = time.time()

                # Update tracking variables
                curr_loss = train_output.loss.item()
                tr_loss += curr_loss
                #self.tr_loss_all.append(curr_loss)
                n_tr_examples += b_input_ids.size(0)
                n_tr_steps += 1

            # training metrics
            loss = tr_loss / n_tr_steps
            self.tr_loss_epoch.append(loss)
            self.metrics_by_epoch["loss"].append(loss)

            if not os.path.exists(out_path):
                os.mkdir(out_path)
            print(f"Saving model to {out_path}...")
            self.model.save_pretrained(os.path.join(out_path, out_epoch_name))

            # validation metrics
            results = self.validate(val_dataloader)
            acc, f1 = results["acc"], results["f1"]
            self.metrics_by_epoch["val_acc"].append(acc)
            self.metrics_by_epoch["val_f1"].append(f1)
            now = file_helpers.get_now_string()
            outf.write(f"\t [{now}] - Train loss: {loss}, Val acc: {acc}, Val F1: {f1}\n")

            # plot progress
            plot_scores(self.metrics_by_epoch, out_path, out_epoch_name)

            if results["f1"] < max_f1:
                trigger_f1 += 1
            else:
                trigger_f1 = 0
                max_f1 = results["f1"]

            if trigger_f1 > 2:
                return

        plot_scores(self.metrics_by_epoch, out_path, out_epoch_name)

    def train(self, train_dataloader, out_model_path=None):
        self.model.train()

        tr_loss = 0
        n_tr_examples, n_tr_steps = 0, 0

        for step, batch in enumerate(train_dataloader):
            if step % 100 == 0:
                n_steps = train_dataloader.batch_sampler.sampler.num_samples // train_dataloader.batch_sampler.batch_size
                print(f"[{file_helpers.get_now_string()}] Batch {step} / {n_steps}")

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

        if out_model_path:
            if not os.path.exists(out_model_path):
                os.mkdir(out_model_path)
                print(out_model_path)
            self.model.save_pretrained(out_model_path)

        return tr_loss, n_tr_examples, n_tr_steps

    def validate(self, validation_dataloader):
        self.model.eval()
        preds = None
        out_label_ids = None

        batch_iterator = tqdm(validation_dataloader, desc="Epochs", disable=False)

        for batch in batch_iterator: #validation_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                # Forward pass
                eval_output = self.model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask)

            logits = eval_output.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            """if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = b_input_ids["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, b_input_ids["labels"].detach().cpu().numpy(), axis=0)"""

        preds = np.argmax(logits, axis=1)

        # Calculate validation metrics
        results = compute_metrics(preds, label_ids)

        return results

def update_tokenizer(tokenizer, new_tokens=["[BOW]", "[EOW]"], out_dir="model/saved"):
    tokenizer.add_tokens(new_tokens)
    model = BertForSequenceClassification.from_pretrained(
            "EMBEDDIA/crosloengual-bert", num_labels=2, output_attentions = False, output_hidden_states = False)
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.save_pretrained(out_dir)

def prepare_data_crossval(train_fname, val_fname, tokenizer, batch_size=32, n=3, mark_word=False, syn=False):
    train_dataloaders = []
    val_dataloaders = []

    for i in [1, 2]: #range(n):
        train_file, val_file = f"{train_fname}minitrain.txt", f"{val_fname}minival.txt" # f"{train_fname}{i}.txt", f"{val_fname}{i}.txt"

        token_id, attention_masks, labels  = get_data(train_file, tokenizer, mark_word=mark_word)
        train_set = TensorDataset(token_id, attention_masks, labels)
        train_dataloader = DataLoader(
            train_set,
            sampler=RandomSampler(train_set),
            batch_size=batch_size
        )
        train_dataloaders.append(train_dataloader)

        token_id, attention_masks, labels  = get_data(val_file, tokenizer)
        val_set = TensorDataset(token_id, attention_masks, labels)
        validation_dataloader = DataLoader(
            val_set,
            sampler=SequentialSampler(val_set),
            batch_size=batch_size
        )
        val_dataloaders.append(validation_dataloader)

    return train_dataloaders, val_dataloaders

def get_data(filename, tokenizer, limit_range=None, shuffle_data=True, mark_word=False):
    data_dict = parse_sentence_data(filename, limit_range, shuffle_lines=shuffle_data)
    data_dict = trim_sentences(data_dict, tokenizer)
    data = add_special_tokens(data_dict, mark_word=mark_word)

    labels = torch.tensor(data_dict['labels'], dtype=torch.long)

    encodings = [tokenizer.encode_plus(" ".join(s), truncation=True, max_length=512, add_special_tokens=True,
                                            padding=True, pad_to_multiple_of=512) for s in data]
    #print("result:", tokenizer.decode(encodings[0]['input_ids']))

    input_ids = torch.tensor([e.get('input_ids') for e in encodings])
    attention_masks = torch.tensor([e.get('attention_mask') for e in encodings])

    return input_ids, attention_masks, labels

def add_special_tokens(data_dict, mark_word=False):
    if mark_word:
        data = []
        for ss, ii in zip(data_dict['sentence_pairs'], data_dict['index_pairs']):
            s1, s2 = ss
            i1, i2 = ii

            s1_split, s2_split = s1.split(" "), s2.split(" ")

            t1_start, t1_end = "<R1>", "</R1>"
            t2_start, t2_end = "<R2>", "</R2>"

            s1_new = s1_split[:i1[0]] + [t1_start] + s1_split[i1[0]: i1[1] + 1] + [t1_end] + s1_split[i1[1] + 1:] + ["[SEP]"]
            s2_new = s2_split[:i2[0]] + [t2_start] + s2_split[i2[0]: i2[1] + 1] + [t2_end] + s2_split[i2[1] + 1:]

            data.append(s1_new + s2_new)
    else:
        #self.data = [f"[CLS] {s1} [SEP] {s2} [SEP]".split(" ") for s1, s2 in self.data_dict['sentence_pairs']]
        data = [f"{s1} [SEP] {s2}".split(" ") for s1, s2 in data_dict['sentence_pairs']]

    return data

def trim_sentences(data_dict, tokenizer):
    word_pairs = data_dict['word_pairs']
    sentence_pairs = data_dict['sentence_pairs']
    index_pairs = data_dict['index_pairs']

    indices_list = [x for i in index_pairs for x in i]
    sentences_list = [x for s in sentence_pairs for x in s]
    words_list = [x for s in word_pairs for x in s]
    n = len(sentences_list)

    for i in range(n):
        new_sentence, _, word_indices = get_token_range(sentences_list[i].split(" "), words_list[i], indices_list[i], tokenizer)
        sentences_list[i] = new_sentence
        indices_list[i] = word_indices

    data_dict['sentence_pairs'] = [(sentences_list[2*i], sentences_list[2*i + 1]) for i in range(n // 2)]
    data_dict['index_pairs'] =  [(indices_list[2*i], indices_list[2*i + 1]) for i in range(n // 2)]

    return data_dict

def find_best(train_filename, val_filename, out_path, lrs=[3e-4], epochs=5, # 5e-5, 3e-5], epochs=5,
              mark_word=False, batch_sizes=[128], syn=False):

    tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/crosloengual-bert")
    if mark_word:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<R1>", "</R1>", "<R2>", "</R2>"]})

    max_accuracy = -1
    idx = -1
    best_model = None

    with open(os.path.join(out_path, "info.txt"), "w", encoding="utf8") as f:
        f.write(out_path + "\n")

        for batch_size in batch_sizes:
            train_dataloaders, val_dataloaders = prepare_data_crossval(train_filename, val_filename, tokenizer,
                                                                       mark_word=mark_word, batch_size=batch_size, n=1, syn=syn)

            for lr in lrs:
                print(f"Using learning rate {lr}")
                f.write(f"\nModels with learning rate {lr}, batch size {batch_size}\n")

                val_f1s = run_crossval_models(train_dataloaders, val_dataloaders, f, lr, epochs, batch_size,
                                                  resize_model=mark_word, tokenizer=tokenizer, out_path=out_path)

                if max([max(x) for x in val_f1s]) > max_accuracy:
                    idx = np.argmax([np.argmax(x) for x in val_f1s])
                    max_accuracy = val_f1s[idx]
                    epoch = np.argmax(val_f1s[idx])
                    best_model = f"train{idx}_{lr}_{batch_size}_{epoch}"

    print(f"\n\nBest model {best_model} at epoch {idx} with val_accuracy: {max_accuracy}")

def score_models(in_folder, test_file, out_file, tokenizer, mark_word=True):
    with open(out_file, "w", encoding="utf8") as f:

        for folder in os.listdir(in_folder):
            f.write(f"\nScore for model {folder}\n")

            model_path = os.path.join(in_folder, folder)
            print(model_path)

            model = BertForSequenceClassification.from_pretrained(model_path)
            model_name = folder

            token_id, attention_masks, labels = get_data(test_file, tokenizer, mark_word=mark_word)
            test_data = TensorDataset(token_id,
                                    attention_masks,
                                    labels)

            test_dataloader = DataLoader(
                test_data,
                sampler=SequentialSampler(test_data),
                batch_size=32
            )

            my_model = AntSynModel(use_tokenizer=tokenizer)
            my_model.model = model

            scores = my_model.validate(test_dataloader)
            f1, acc = scores["f1"], scores["acc"]

            f.write(f"F1 score: {f1}, acc: {acc}\n")


def run_crossval_models(train_dataloaders, val_dataloaders, f, lr, n_epochs, batch_size, resize_model=False,
                        tokenizer=None, out_path=None):
    val_f1s = []
    i = 0
    for train_dataloader, val_dataloader in zip(train_dataloaders, val_dataloaders):
        print(f"Model #{i}")
        f.write(f"Model #{i}\n")

        model_name = f"train{i}_{lr}_{batch_size}_{i}"

        model = AntSynModel(resize_model=resize_model, use_tokenizer=tokenizer)
        model.finetune_2(val_dataloader, train_dataloader, f, lr=lr, epochs=n_epochs, out_path=out_path, out_name=model_name)

        val_f1s.append(model.metrics_by_epoch["val_f1"])
        i += 1

    return val_f1s
