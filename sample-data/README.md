# Sample data
This folder gives introduction to the data format for the generated data by running a quick example:
```bash
bash sample-data.sh
```
And this will generated the data into [./generated/fiqa](./generated/fiqa):

```bash
.
├── corpus.jsonl  # Copied from the target dataset fiqa/
├── qgen-qrels  # Synthetic query generation. BeIR format.
│   └── train.tsv  # Each line: query_id \t passage_id \t label (binary)
├── qgen-queries.jsonl  # Synthetic query generation. BeIR format. Mapping from query IDs to query texts.
├── hard-negatives.jsonl  # Negative mining. Mapping from query IDs to positive-passage IDs and negative-passages IDs.
└── gpl-training-data.tsv  # Pseudo labeling. Final data, ready for GPL training. Each line: query_id \t positive_passage_id \t negative_passage_id \t ce_score
```

The prefix "qgen-" is actually specified by the argument `--qgen_prefix`.
