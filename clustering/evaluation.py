import os
import numpy as np


def get_results(data_file: str):
    results = []

    with open(data_file, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue

            data = line.strip().split("\t")
            word = data[0]
            matrix = parse_confusion_matrix(data[6])

            if matrix is not None:
                correct = np.trace(matrix)
                false = np.sum(matrix) - correct
                results.append([word, correct, false])

    with open(os.path.join(os.path.dirname(data_file), "clusters_eval.tsv"), "w", encoding="utf8") as outf:
        correct = [x for x in results if x[2] == 0]
        incorrect = [x for x in results if x[2] > 0]
        outf.write("Correctly grouped: %d - %s\n" % (len(correct), ",".join([x[0] for x in correct])))
        outf.write("Incorrectly grouped: %d - %s\n" % (len(incorrect), ",".join([x[0] for x in incorrect])))

        close = [x for x in results if x[2] == 0 or x[2] / (x[1] + x[2]) < 0.1]
        outf.write("Close to correct: %d - %s\n" % (len(close), ",".join([x[0] for x in close])))

        loss = results
        loss.sort(key=lambda x: x[2])
        outf.writelines(["\t".join([str(y) for y in x]) + "\n" for x in loss])

    return correct, close, incorrect

def parse_confusion_matrix(confusion_matrix):
    matrix = []

    for x in confusion_matrix.split(","):
        new_row = []
        row = x.strip("[] ")

        for y in row.split(" "):
            if y is None or y == "None":
                return None
            elif y == '':
                pass
            else:
                new_row.append(int(y))

        matrix.append(new_row)

    return np.asmatrix(np.array(matrix))

def find_best_clusters(file_list):
    correct, close, incorrect = [], [], []

    for file in file_list:
        x, y, z = get_results(file)

        correct.append(set([xx[0] for xx in x]))
        close.append(set([yy[0] for yy in y]))
        incorrect.append(set([zz[0] for zz in z]))

    correct_common = set.intersection(*correct)
    close_common = set.intersection(*close)
    print("Correct - common: %d words: %s" % (len(correct_common), ", ".join(correct_common)))
    print("Close - common: %d words: %s" % (len(close_common), ", ".join(close_common)))
    print("#words: %s" % ", ".join([str(len(x) + len(y)) for x, y in zip(correct, incorrect)]))
