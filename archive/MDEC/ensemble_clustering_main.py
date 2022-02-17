import file_helpers
import matlab.engine


def mdec_ensemble_clustering(data_file, words_json_file, out_file):
    words_json = file_helpers.load_json_word_data(words_json_file)
    print("words json loaded")

    eng = matlab.engine.start_matlab()
    n_words = len(words_json.keys())
    idx = 0

    with open(out_file, "w", encoding="utf8") as outf:
        for i in range(len(words_json.keys())):

            words, word_forms, sentences, embeddings, idx = file_helpers.load_sentences_embeddings_file_grouped(
                data_file, start=idx)

            assert len(set(words)) == 1

            if words[0] == "iztočnica":
                continue

            word = words[0]
            print("parsed data: %s\t%d/%d" % (word, i, n_words))

            n_samples = len(sentences)
            n = len(words_json[word])

            if n == 1:
                pass
                results = list(zip([0] * n_samples, words, sentences))
                file_helpers.write_grouped_data(outf, results)

            elif n_samples < n:
                print("Too little samples for %s: %d samples, %d partitions." % (word, n_samples, n))

            else:
                n_clusters = min(n, n_samples)
                #correct_format_embeddings = matlab.double([[0.021,0.2,0.3], [0.022,0.2,0.3], [0.1,0.5,0.3],[.1,.8,.3],[.6,.2,.3], [0.02,0.2,0.3], [0.11,0.5,0.3],[.11,.8,.3],[.61,.2,.3]])

                correct_format_embeddings = matlab.double([e.tolist() for e in embeddings])

                result_MDEC_HC, result_MDEC_SC, result_MDEC_BG = eng.runMDEC(correct_format_embeddings, n_clusters, nargout=3)
                #print(result_MDEC_HC)
                #print(result_MDEC_SC)
                #print(result_MDEC_BG)

                results = list(zip([int(x[0]) for x in result_MDEC_BG], words, sentences))
                results.sort(key=lambda x: x[0])

                file_helpers.write_grouped_data(outf, results) #, centroids=clusterer.cluster_centers_)

    eng.exit()


def mdec_ensemble_clustering_v2(data_file, words_json_file, out_file):
    words_json = file_helpers.load_json_word_data(words_json_file)
    print("words json loaded")

    eng = matlab.engine.start_matlab()
    n_words = len(words_json.keys())
    idx = 0

    with open(out_file, "w", encoding="utf8") as outf:
        for i in range(len(words_json.keys())):

            words, sentences, embeddings, idx = file_helpers.load_sentences_embeddings_file_grouped(
                data_file, words_json.keys(), start=idx, v2=True)

            assert len(set(words)) == 1

            if words[0] == "iztočnica":
                continue

            word = words[0]
            print("parsed data: %s\t%d/%d" % (word, i, n_words))

            n_samples = len(sentences)
            n = len(words_json[word])

            if n == 1:
                pass
                results = list(zip([0] * n_samples, words, sentences))
                file_helpers.write_grouped_data(outf, results)

            elif n_samples < n:
                print("Too little samples for %s: %d samples, %d partitions." % (word, n_samples, n))

            else:
                n_clusters = min(n, n_samples)
                # correct_format_embeddings = matlab.double([[0.021,0.2,0.3], [0.022,0.2,0.3], [0.1,0.5,0.3],[.1,.8,.3],[.6,.2,.3], [0.02,0.2,0.3], [0.11,0.5,0.3],[.11,.8,.3],[.61,.2,.3]])

                correct_format_embeddings = matlab.double([e.tolist() for e in embeddings])

                result_MDEC_HC, result_MDEC_SC, result_MDEC_BG = eng.runMDEC(correct_format_embeddings, n_clusters,
                                                                             nargout=3)
                # print(result_MDEC_HC)
                # print(result_MDEC_SC)
                # print(result_MDEC_BG)

                results = list(zip([int(x[0]) for x in result_MDEC_BG], words, sentences))
                results.sort(key=lambda x: x[0])

                file_helpers.write_grouped_data(outf, results)  # , centroids=clusterer.cluster_centers_)

    eng.exit()


if __name__ == '__main__':
    word_embeddings = "../../data/embeddings/v2/sample_sentences_100.txt"
    word_json_file = "../../data/sources/besede_s_pomeni.json"
    mdec_ensemble_clustering_v2(word_embeddings, word_json_file, "./v2/test.txt")
