# GPL Toolkit
The GPL toolkit contains source files that can both be imported as a module to serve the main entry method `gpl.train` or be runable itself as an independent util.

## reformat
In some cases, a checkpoint cannot be directly loadable (via `--base_ckpt`) by SBERT (or in a correct way), e.g. "facebook/dpr-question_encoder-single-nq-base" and "princeton-nlp/sup-simcse-bert-base-uncased". 

This is because:
1. They are **not in SBERT-format** but in Hugginface-format;
2. And **for Huggingface-format**, SBERT can only work with the checkpoint with **a Transformer layer as the last layer**, i.e. the outputs are hidden states with shape `(batch_size, sequence_length, hidden_dimenstion)`.

However, the last layer of "facebook/dpr-question_encoder-single-nq-base" is actually a dense linear (with an activation function).

To start from these checkpoints, one need to transform them into SBERT-format. We here provide two examples in [reformat.py](./reformat.py) to transform `simcse_like` and `dpr_like` models into SBERT-format. For example, one can run:
```bash
python -m gpl.toolkit.reformat \
    --template "dpr_like" \
    --model_name_or_path "facebook/dpr-question_encoder-single-nq-base" \
    --output_path "dpr-SBERT-format"
```
And then use the reformatted checkpoint saved in the `--output_path`.

What this [`gpl.toolkit.reformat.dpr_lik`](https://github.com/UKPLab/gpl/blob/7272222f290dbdc5e4a7f32be070496f05ffaad8/gpl/toolkit/reformat.py#L30) does is:
1. Load the checkpoint using Huggingface's `AutoModel` into `word_embedding_model`;
2. Then within the `word_embedding_model`, it traces along the path `DPRQuestionEncoder` -> `DPREncoder` -> `BertModel` -> `BertPooler` to get the `BertModel` and the `BertPooler` (the final linear layer of DPR models);
3. Compose everything (including the `BertModel`, a CLS pooling layer, the `BertPooler`) together again into a SBERT-format checkpoint.
4. Save the reformatted checkpoint into the `--output_path`.

## evaluation
We can both evaluate a checkpoint 

1. within the `gpl.train` workflow or 
2. in an independent routine. 
    
For the case (2), we can simply run this below, with the example of SciFact (of course, writing a new Python script with `from gpl.toolkit import evaluate; evaluate(...)` is also supported):
```bash
export dataset="scifact"
if [ ! -d "$dataset" ]; then
    wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$dataset.zip
    unzip $dataset.zip
fi

python -m gpl.toolkit.evaluation \
    --data_path $dataset \
    --output_dir $dataset \
    --model_name_or_path "GPL/msmarco-distilbert-margin-mse" \
    --max_seq_length 350 \
    --score_function "dot" \
    --sep " " \
    --k_values 10 100
```
This will save the results in `--output_dir`:
```bash
# cat scifact/results.json | json_pp
{
   "map" : {
      "MAP@10" : 0.53105,
      "MAP@100" : 0.53899
   },
   "mrr" : {
      "MRR@10" : 0.54623
   },
   "ndcg" : {
      "NDCG@10" : 0.57078,
      "NDCG@100" : 0.60891
   },
   "precicion" : {
      "P@10" : 0.07667,
      "P@100" : 0.0097
   },
   "recall" : {
      "Recall@10" : 0.6765,
      "Recall@100" : 0.85533
   }
}
```
