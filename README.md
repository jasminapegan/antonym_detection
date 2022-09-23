# Semantic detection of synonyms and antonyms with contextual embeddings
## Pomenska detekcija sopomenk in protipomenk s kontekstualnimi vložitvami

This project contains work on MSc thesis concerning detection of synonyms and antonyms of given word sense.

Contents:
- `data` Functions for parsing and processing data, including creating word embeddings.
- `clustering` Functions for clustering part, including cluster visualization and finding synonym/antonym clusters.
- `classification` Functions for classification with BERT and R-BERT models
 * `classifier` R-BERT implementation, source [https://github.com/monologg/R-BERT][https://github.com/monologg/R-BERT]
- `file_helpers.py` General methods used in multiple parts of the project.