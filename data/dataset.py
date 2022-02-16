import file_helpers


def create_val_test_set(in_data, given_data, val_file, test_file, ratio=0.5, tmp_dir="tmp/"):
    intersection = tmp_dir + "intersection.txt"
    difference = tmp_dir + "difference.txt"

    file_helpers.filter_file_by_words(in_data, given_data, intersection, skip_idx=1)    # skip classification
    file_helpers.filter_file_by_words(in_data, given_data, difference, skip_idx=1, complement=True)

    pt1 = tmp_dir + "pt1.txt"
    get_random_part(intersection, pt1, test_file, ratio=ratio)

    file_helpers.concatenate_files([pt1, difference], val_file)


def get_random_part(in_file, out_file1, out_file2, ratio=0.5):
    n = file_helpers.file_len(in_file)
    n_part = n * ratio

    words_data = file_helpers.load_validation_file_grouped(in_file, all_strings=True)
    words_count = [(key, len(words_data[key]['sentences'])) for key in words_data.keys()]
    file_helpers.shuffle(words_count)

    sum = 0
    idx = 0
    words_part = []

    while sum < n_part:
        word, count = words_count[idx]
        words_part.append(word)
        sum += count
        idx += 1

    print("File1: %d, File2: %d, total: %d" % (sum, n-sum, n))

    with open(out_file1, "w", encoding="utf8") as out1:
        with open(out_file2, "w", encoding="utf8") as out2:
            with open(in_file, "r", encoding="utf8") as f:

                for line in f:
                    word = line.split("\t")[0]

                    if word in words_part:
                        out1.write(line)
                    else:
                        out2.write(line)
