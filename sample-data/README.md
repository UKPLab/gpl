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

Let us now have a detailed look at format:

1. corpus.json
    ```bash
    # head -n 1 corpus.jsonl | json_pp
    {
    "_id" : "460230",
    "text" : "Market capitalization is one way to represent the value of the company. So if a company has 10 million shares, which are each worth $100, then the company's market capitalization is 1 billion. Large cap companies tend to be larger and more stable. Small cap companies are smaller, which indicates higher volatility. So if you want more aggressive investments then you may want to invest in small cap companies while if you lean on the side of caution then big cap companies may be your friend.",
    "title" : ""
    }
    ```
2. qgen-qrels/train.tsv
    ```bash
    # head -n 4 qgen-qrels/train.tsv
    query-id	corpus-id	score
    genQ1	460230	1
    genQ2	460230	1
    genQ3	460230	1
    ```
3. qgen-queries.jsonl
    ```bash
    # head -n 1 qgen-queries.jsonl | json_pp
    {
    "_id" : "genQ1",
    "metadata" : {},  # not used in GPL
    "text" : "what is the difference between large cap companies and smaller cap"
    }
    ```
4. hard-negatives.jsonl
    ```bash
    # head -n 1 hard-negatives.jsonl | json_pp
    {
    "neg" : {
        "msmarco-MiniLM-L-6-v3" : [
            "281423",  # each one here is a negative-candidate ID
            "316535",  # the rank actually does not matter, since GPL uses random sampling
            "257122",
            "78297",
            "511432",
            "182744",
            "35856",
            "120306",
            "214079"
        ],
        "msmarco-distilbert-base-v3" : [
            "281423",
            "257122",
            "316535",
            "35856",
            "511432",
            "182744",
            "120306",
            "78297",
            "214079"
        ]
    },
    "pos" : [
        "460230"  # this passage with ID "460230" was used to generate query with ID "genQ1". And this pair is viewed as a positive example
    ],
    "qid" : "genQ1"
    }
    ```
5. gpl-training-data.tsv
    ```bash
    # head -n 3 gpl-training-data.tsv 
    # query ID \t positive-passage ID \t negative-passage ID \t margin, i.e. CE(query, positive) - CE(query, negative)
    genQ5	257122	460230	18.465141
    genQ19	511432	257122	15.421183
    genQ26	182744	214079	19.373644
    ```