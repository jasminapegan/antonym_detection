from sklearn.metrics import f1_score
import os
from matplotlib import pyplot as plt


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
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(preds, labels)
    print("acc, f1", acc, f1)
    return {
        "acc": acc,
        "f1": f1
    }

def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)