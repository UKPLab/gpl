# GPL Toolkit
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
