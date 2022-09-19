from sklearn.metrics import f1_score, precision_score, recall_score
import os
from matplotlib import pyplot as plt
import numpy as np

from classification.dataset import parse_sentence_data


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

def simple_accuracy(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    return (preds == labels).mean()

def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(preds, labels)
    prec = precision_score(preds, labels)
    rec = recall_score(preds, labels)
    print("acc, f1", acc, f1)
    return {
        "acc": acc,
        "f1": f1,
        "prec": prec,
        "rec": rec
    }

def get_sentence_predictions(in_file, test_file, out_file):
    data = parse_sentence_data(test_file)
    word_pairs = data['word_pairs']
    sentence_pairs = data['sentence_pairs']
    labels = data['labels']

    with open(in_file, "r", encoding="utf8") as f:
        preds = [x.strip().split("\t") for x in f.readlines()]

    with open(out_file, "w", encoding="utf8") as outf:
        print(len(word_pairs), len(preds))
        #assert len(word_pairs) == len(preds)
        for wp, sp, label, pred_data in zip(word_pairs, sentence_pairs, labels, preds):
            w1, w2 = wp
            s1, s2 = sp
            _, _, pred = pred_data
            outf.write(f"{w1}\t{w2}\t{s1}\t{s2}\t{label}\t{pred}\n")

def split_pred_file(in_file, out_dir):
    with open(in_file, "r", encoding="utf8") as f:
        data = [line.strip().split("\t") for line in f.readlines()]

    words = list(set([d[0] for d in data] + [d[1] for d in data]))
    words.sort()

    split_data = {}

    split_data["tp"] = [x for x in data if x[-1] == "1" and x[-2] == "1"]
    split_data["tn"] = [x for x in data if x[-1] == "0" and x[-2] == "0"]
    split_data["fp"] = [x for x in data if x[-1] == "0" and x[-2] == "1"]
    split_data["fn"] = [x for x in data if x[-1] == "1" and x[-2] == "0"]

    for k in split_data.keys():
        with open(f"{out_dir}/proposed_{k}", "w", encoding="utf8") as f:
            word_counts = {w: 0 for w in words}

            for line in split_data[k]:
                f.write("\t".join(line) + "\n")

                w1, w2 = line[:2]
                word_counts[w1] += 1
                word_counts[w2] += 1

            f.write("\nWord counts\n")
            for w in words:
                f.write(f"{w}: {word_counts[w]}\n")

split_pred_file("best/ant/ant_proposed_sentences.txt", "best/ant")
split_pred_file("best/syn/syn_proposed_sentences.txt", "best/syn")

