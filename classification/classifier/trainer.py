import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import BertConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW

from classification.classifier.model import RelationModel
from classification.classifier.utils import compute_metrics, get_label, write_prediction, plot_scores

logging.basicConfig(filename="logs.txt",
                    filemode="a",
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)



class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.label_lst = get_label(args)
        self.num_labels = len(self.label_lst)

        self.config = BertConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=self.num_labels,
            finetuning_task=args.task,
            id2label={str(i): label for i, label in enumerate(self.label_lst)},
            label2id={label: i for i, label in enumerate(self.label_lst)},
        )
        self.model = RelationModel.from_pretrained(args.model_name_or_path, config=self.config, args=args)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self, out_filename):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.args.train_batch_size,
        )

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=t_total,
        )

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Train data = %s", (self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Learning rate = %d", self.args.learning_rate)
        logger.info("  Number of layers = %d", self.args.n_layers)
        logger.info("  Layer size divisor = %d", self.args.layer_size_divisor)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Save epochs = %d", self.args.save_epochs)
        logger.info("  Min epochs before early stopping = %d", self.args.early_stopping_epochs)
        logger.info("  Patience for early stopping = %d", self.args.early_stopping_patience)

        global_step = 0
        tr_loss = 0.0
        trigger_f1 = 0
        max_f1 = 0.0
        metrics_by_epoch = {"loss": [],
                            #"val_loss": [],
                            "val_acc": [],
                            "val_f1": []}
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for i, _ in enumerate(train_iterator):
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
            epoch_steps = 0
            epoch_loss = 0

            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                    "e1_mask": batch[4],
                    "e2_mask": batch[5],
                }
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                epoch_loss += loss.item()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1
                    epoch_steps += 1

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            curr_loss = epoch_loss / epoch_steps
            metrics_by_epoch["loss"].append(curr_loss)
            results = self.evaluate("test")
            #metrics_by_epoch["val_loss"].append(results["loss"])
            metrics_by_epoch["val_acc"].append(results["acc"])
            metrics_by_epoch["val_f1"].append(results["f1"])

            do_save = False
            if results["f1"] < max_f1:
                trigger_f1 += 1
            else:
                trigger_f1 = 0
                max_f1 = results["f1"]
                do_save = True

            if i > self.args.early_stopping_epochs:
                if trigger_f1 > self.args.early_stopping_patience:
                    logger.info(f"  Early stopping after {i} epochs!")
                    self.save_model(out_filename, epoch=i)
                    train_iterator.close()
                    break

            if self.args.save_epochs > 0 and global_step % self.args.save_epochs == 0 or do_save:
                self.save_model(out_filename, epoch=i)

                try:
                    plot_scores(metrics_by_epoch, self.args.eval_dir, out_filename)
                except:
                    pass

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        self.save_model(out_filename)

        return global_step, tr_loss / global_step, metrics_by_epoch

    def evaluate(self, mode):
        # We use test dataset because semeval doesn't have dev dataset
        if mode == "test":
            dataset = self.test_dataset
        elif mode == "dev":
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=False):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                    "e1_mask": batch[4],
                    "e2_mask": batch[5],
                }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                curr_loss = tmp_eval_loss.mean().item()
                eval_loss += curr_loss
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {"loss": eval_loss}
        preds_bin = np.argmax(preds, axis=1)
        write_prediction(self.args, os.path.join(self.args.eval_dir, "proposed_answers.txt"), preds, preds_bin)

        result = compute_metrics(preds_bin, out_label_ids)
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  {} = {:.4f}".format(key, results[key]))

        return results

    def save_model(self, filename, epoch=-1):
        model_dir = self.args.model_dir
        model_dir = os.path.join(model_dir, f"{filename}_{epoch}")

        # Save model checkpoint (Overwrite)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(model_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", model_dir)

    def load_model(self, filename):
        model_dir = os.path.join(self.args.model_dir, filename)
        # Check whether model exists
        if not os.path.exists(model_dir):
            print(model_dir)
            raise Exception("Model doesn't exists! Train first!")

        self.args = torch.load(os.path.join(model_dir, "training_args.bin"))
        self.model = RelationModel.from_pretrained(model_dir, args=self.args)
        self.model.to(self.device)
        logger.info("***** Model Loaded *****")