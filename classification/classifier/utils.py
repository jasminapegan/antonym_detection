import logging
import os
import random

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

from matplotlib import pyplot as plt

ADDITIONAL_SPECIAL_TOKENS = ["<R1>", "</R1>", "<R2>", "</R2>"]


def get_label(args):
    return ['0', '1'] #[label.strip() for label in open(os.path.join(args.data_dir, args.label_file), "r", encoding="utf-8")]


def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


def write_prediction(args, output_file, preds, preds_bin):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    relation_labels = get_label(args)
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\t{}\n".format(idx, relation_labels[pred], preds_bin))


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(preds, labels)
    print("acc, f1", acc, f1)
    return {
        "acc": acc,
        "f1": f1
    }

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n+1)

def plot_scores(metrics, out_dir, model_name):
    print(f"Plotting scores for model {model_name}...")

    metric_names = sorted(list(metrics.keys()))
    epochs = range(1, len(metrics[metric_names[0]]) + 1)
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    plt.subplot(2, 1, 1)
    cmap = get_cmap(len(metric_names))

    for i, name in enumerate(metric_names):
        plt.plot(epochs, metrics[name], c=cmap(i), label=name)

    plt.title(f'Metrics for model {model_name}')
    plt.xlabel('Epochs')
    plt.legend(loc='lower right')

    fname = os.path.join(out_dir, f"plot_{model_name}.png")
    plt.savefig(fname)
    plt.close(fig)
