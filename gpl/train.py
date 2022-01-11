from beir.datasets.data_loader import GenericDataLoader
from .toolkit import qgen, NegativeMiner, MarginDistillationLoss, GenerativePseudoLabelingDataset, PseudoLabeler, evaluate, resize, load_sbert, set_logger_format
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
import os
import logging
import argparse
from typing import List
# import crash_ipdb


set_logger_format()
logger = logging.getLogger('gpl.train')  # Here we do not use __name__ to have unified logger name, no matter whether we are using `python -m` or `import gpl; gpl.train`


def train(
    path_to_generated_data: str,
    output_dir: str,
    do_evaluation: str = False,
    evaluation_data: str = None,
    evaluation_output: str = 'output',
    qgen_prefix: str = 'qgen',
    base_ckpt: str = 'distilbert-base-uncased',
    generator: str = 'BeIR/query-gen-msmarco-t5-base-v1',
    cross_encoder: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    batch_size_gpl: int = 32,
    batch_size_generation: int = 32,
    pooling: str = None,
    max_seq_length: int = 350,
    new_size: int = None,
    queries_per_passage: int = 3,
    gpl_steps: int = 140000,
    use_amp: bool = False,
    retrievers: List[str] = ['msmarco-distilbert-base-v3', 'msmarco-MiniLM-L-6-v3'],
    negatives_per_query: int = 50,
    sep:str = ' ',
    k_values: List[int] = [10,]
):     
    #### Assertions ####
    assert pooling in [None, 'mean', 'cls', 'max']
    if do_evaluation:
        assert evaluation_data is not None
        assert evaluation_output is not None
        try:
            GenericDataLoader(evaluation_data)
        except Exception as e:
            logger.error('Cannot load evaluation data for evaluation usage.')
            raise e
    if new_size is not None:
        assert new_size * queries_per_passage >= batch_size_gpl


    #### Make sure there is a `corpus.jsonl` file. It should be under either `path_to_generated_data` or `evaluation_data`` ####
    #### Also resize the corpus for efficient training if required  ####
    os.makedirs(path_to_generated_data, exist_ok=True)
    if 'corpus.jsonl' not in os.listdir(path_to_generated_data):
        logger.info(f'Corpus does not exist in {path_to_generated_data}. Now clone the one from the evaluation path {evaluation_data}')
        assert 'corpus.jsonl' in os.listdir(evaluation_data), f'No corpus found in evaluation path {evaluation_data}! It should be in the BeIR format. For more details, please refer to https://github.com/UKPLab/beir#beers-available-datasets.'
        if new_size is not None:
            resize(evaluation_data, path_to_generated_data, new_size)
        else:
            corpus_path = os.path.join(evaluation_data, 'corpus.jsonl')
            os.system(f'cp {corpus_path} {path_to_generated_data}')
    

    #### Synthetic query generation ####
    #### This will be skipped if there is an existing `gen-queries.jsonl`file under `path_to_generated_data` ####
    if f'{qgen_prefix}-qrels' in os.listdir(path_to_generated_data) and f'{qgen_prefix}-queries.jsonl' in os.listdir(path_to_generated_data):
        logger.info('Loading from existing generated data')
        corpus, gen_queries, gen_qrels = GenericDataLoader(path_to_generated_data, prefix=qgen_prefix).load(split="train")
    else:
        logger.info('No generated queries found. Now generating it')
        assert 'corpus.jsonl' in os.listdir(path_to_generated_data), 'At least corpus should exist!'
        qgen(path_to_generated_data, path_to_generated_data, generator_name_or_path=generator, ques_per_passage=queries_per_passage, bsz=batch_size_generation, qgen_prefix=qgen_prefix)
        corpus, gen_queries, gen_qrels = GenericDataLoader(path_to_generated_data, prefix=qgen_prefix).load(split="train")


    #### Hard-negative mining ####
    #### This will be skipped if there is an existing `hard-negatives.jsonl` file under `path_to_generated_data` ####
    if 'hard-negatives.jsonl' in os.listdir(path_to_generated_data):
        logger.info('Using exisiting hard-negative data')
    else: 
        logger.info('No hard-negative data found. Now mining it')
        miner = NegativeMiner(path_to_generated_data, qgen_prefix, retrievers=retrievers, nneg=negatives_per_query, sep=sep)
        miner.run()


    #### Pseudo labeling ####
    #### This will be skipped if there is an existing `gpl-training-data.tsv` file under `path_to_generated_data` ####
    if 'gpl-training-data.tsv' in os.listdir(path_to_generated_data):
        logger.info('Using existing GPL-training data')
    else:
        logger.info('No GPL-training data found. Now generating it via pseudo labeling')
        pseudo_labeler = PseudoLabeler(path_to_generated_data, gen_queries, corpus, gpl_steps, batch_size_gpl, cross_encoder, max_seq_length)
        pseudo_labeler.run()
    

    ### Train the model with MarginMSE loss ###
    #### This will be skipped if the checkpoint at the indicated training steps can be found ####
    ckpt_dir = os.path.join(output_dir, str(gpl_steps))
    if not os.path.exists(ckpt_dir) or (os.path.exists(ckpt_dir) and not os.listdir(ckpt_dir)):
        logger.info('Now doing training on the generated data with the MarginMSE loss')
        #### It can load checkpoints in both SBERT-format (recommended) and Huggingface-format
        model: SentenceTransformer = load_sbert(base_ckpt, pooling, max_seq_length)

        fpath_gpl_data = os.path.join(path_to_generated_data, 'gpl-training-data.tsv')
        train_dataset = GenerativePseudoLabelingDataset(fpath_gpl_data, gen_queries, corpus, sep)
        train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size_gpl, drop_last=True)  # Here shuffle=False, since (or assuming) we have done it in the pseudo labeling
        train_loss = MarginDistillationLoss(model=model)

        # assert gpl_steps > 1000
        model.fit(
            [(train_dataloader, train_loss),],
            epochs=1,
            steps_per_epoch=gpl_steps,
            warmup_steps=1000,
            checkpoint_save_steps=10000,
            checkpoint_save_total_limit=10000,
            output_path=output_dir,
            checkpoint_path=output_dir,
            use_amp=use_amp
        )
    else:
        logger.info('Trained GPL model found. Now skip training')


    ### Evaluate the model if required ###
    if do_evaluation:
        logger.info('Doing evaluation for GPL')
        evaluate(
            evaluation_data, 
            evaluation_output, 
            ckpt_dir, 
            max_seq_length,
            score_function='dot',  # Since for now MarginMSE only works with dot-product
            pooling=pooling,
            sep=sep,
            k_values=k_values
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_generated_data', required=True, help='Path for/to the generated data. GPL will first check this path for a `corpus.jsonl` file for the (sole) data input of the whole pipeline. If an empty folder is indicated, query generation and hard-negative mining will be run automatically; one can also use a BeIR-QGen format data folder to start and skip the query generation.')
    parser.add_argument('--output_dir', required=True, help='Output path for the GPL model.')
    parser.add_argument('--do_evaluation', action='store_true', default=False, help='Wether to do the evaluation (after training)')
    parser.add_argument('--evaluation_data', type=str, help='Path to the BeIR-format dataset. This is the next folder GPL goes to for the target corpus if there is no `corpus.jsonl` under `path_to_generated_data`')
    parser.add_argument('--evaluation_output', default='output', help='Path for the evaluation output.')
    parser.add_argument('--qgen_prefix', default='qgen', help='This prefix will appear as part of the (folder/file) names for query-generation results: For example, we will have "gen-qrels/" and "gen-queries.jsonl" by default.')
    parser.add_argument('--base_ckpt', default='distilbert-base-uncased', help='Initialization checkpoint in HF or SBERT format. Meaning-pooling will be used.')
    parser.add_argument('--generator', default='BeIR/query-gen-msmarco-t5-base-v1')
    parser.add_argument('--cross_encoder', default='cross-encoder/ms-marco-MiniLM-L-6-v2')
    parser.add_argument('--batch_size_gpl', type=int, default=32)
    parser.add_argument('--batch_size_generation', type=int, default=32)
    parser.add_argument('--pooling', type=str, default=None, choices=['cls', 'mean', 'max'], help="Specifying pooling method for dense retriever if in Huggingface-format. By default (None), it uses mean pooling. If in SBERT-format, there would be the indicated pooling method in its configure file and thus this argument will be ignored. ")
    parser.add_argument('--max_seq_length', type=int, default=350)
    parser.add_argument('--new_size', type=int, default=None, help='Resize the corpus to `new_size` if needed.')
    parser.add_argument('--queries_per_passage', type=int, default=3)
    parser.add_argument('--gpl_steps', type=int, default=140000, help='Training steps for GPL.')
    parser.add_argument('--use_amp', action='store_true', default=False, help='Whether to use half precision')
    parser.add_argument('--retrievers', nargs='+', default=['msmarco-distilbert-base-v3', 'msmarco-MiniLM-L-6-v3'], help='Indicate retriever names. They could be one or many BM25 ("bm25") or dense retrievers (in SBERT format).')
    parser.add_argument('--negatives_per_query', type=int, default=50, help="Mine how many negatives per query per retriever")
    parser.add_argument('--sep', type=str, default=' ', help="Separation token between title and body text for each passage. The concatenation way is `sep.join([title, body])`")
    parser.add_argument('--k_values', nargs='+', type=int, default=[10,], help="The K values in the evaluation. This will compute nDCG@K, recall@K, precision@K and MAP@K")
    args = parser.parse_args()
    train(**vars(args))