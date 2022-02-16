def prepare_data(in_file, out_file, labels_file):

    with open(in_file, "r", encoding="utf8") as f:
        with open(out_file, "w", encoding="utf8") as out_f:
            with open(labels_file, "w", encoding="utf8") as labels_f:

                labels_f.write("label\tword\tsentence\n")

                for line in f:
                    label, word, sentence, embedding = line.strip().split("\t")

                    out_f.write(embedding.replace(" ", "\t") + "\n")
                    labels_f.write("\t".join([label, word, sentence]) + "\n")

