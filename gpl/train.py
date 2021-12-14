from random import choice
from ast import parse
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.generation import QueryGenerator as QGen
from beir.generation.models import QGenModel
from beir.retrieval.train import TrainRetriever
from sentence_transformers import SentenceTransformer, losses, models
from .toolkit import qgen, NegativeMiner, MarginDistillationLoss, GenerativePseudoLabelingDataset, PseudoLabeler, evaluate, resize, mnrl
from torch.utils.data import DataLoader

import pathlib, os
import logging

import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

parser = argparse.ArgumentParser()
parser.add_argument('--generated_path', required=True, help='Path for/to the generated data. If an empty folder is indicated, query generation and hard-negative mining will be run automatically (the corpus.jsonl file in `evaluation_data` will be used!!!); one can also use a BeIR-QGen format data folder to start and skip the query generation.')
parser.add_argument('--base_ckpt', default='distilbert-base-uncased', help='Initialization checkpoint in HF or SBERT format. Meaning-pooling will be used.')
parser.add_argument('--generator', default='BeIR/query-gen-msmarco-t5-base-v1')
parser.add_argument('--cross_encoder', default='cross-encoder/ms-marco-MiniLM-L-6-v2')
parser.add_argument('--batch_size_gpl', type=int, default=32)
parser.add_argument('--batch_size_generation', type=int, default=32)
parser.add_argument('--pooling', type=str, default='mean', choices=['cls', 'mean', 'max'])
parser.add_argument('--max_seq_length', type=int, default=350)
parser.add_argument('--new_size', type=int, default=None, help='Resize the corpus to `new_size` if needed.')
parser.add_argument('--queries_per_passage', type=int, default=3)
parser.add_argument('--mnrl_output_dir', default=None, help='If specified, QGen will be trained on the same data and saved to this path.')
parser.add_argument('--output_dir', required=True, help='Output path for the GPL model.')
parser.add_argument('--evaluation_data', required=True, help='Path to the BeIR-format dataset.')
parser.add_argument('--evaluation_output', required=True, help='Path for the evaluation output.')
parser.add_argument('--mnrl_evaluation_output', help='Path for the QGen evaluation output.')
parser.add_argument('--gpl_steps', type=int, default=140000, help='Training steps for GPL.')
parser.add_argument('--use_amp', action='store_true', default=False, help='Whether to use half precision')
parser.add_argument('--retrievers', nargs='+', default=['msmarco-distilbert-base-v3', 'msmarco-MiniLM-L-6-v3'], help='Indicate retriever names. They could be one or many BM25 ("bm25") or dense retrievers (in SBERT format).')
args = parser.parse_args()

prefix = "gen"
#### Training on Generated Queries ####
os.makedirs(args.generated_path, exist_ok=True)
if 'corpus.jsonl' not in os.listdir(args.generated_path):
    logger.info(f'Corpus does not exist in {args.generated_path}. Now clone the one from the evaluation path {args.evaluation_data}')
    assert 'corpus.jsonl' in os.listdir(args.evaluation_data), f'No corpus found in evaluation path {args.evaluation_data}!'
    if args.new_size is not None:
        resize(args.evaluation_data, args.generated_path, args.new_size)
    else:
        corpus_path = os.path.join(args.evaluation_data, 'corpus.jsonl')
        os.system(f'cp {corpus_path} {args.generated_path}')

if 'gen-qrels' in os.listdir(args.generated_path) and 'gen-queries.jsonl' in os.listdir(args.generated_path):
    logger.info('Loading from existing generated data')
    corpus, gen_queries, gen_qrels = GenericDataLoader(args.generated_path, prefix=prefix).load(split="train")
else:
    logger.info('No generated data found. Now generating it')
    assert 'corpus.jsonl' in os.listdir(args.generated_path), 'At least corpus should exist!'
    qgen(args.generated_path, args.generated_path, generator_name_or_path=args.generator, ques_per_passage=args.queries_per_passage, bsz=args.batch_size_generation)
    corpus, gen_queries, gen_qrels = GenericDataLoader(args.generated_path, prefix=prefix).load(split="train")

if 'hard-negatives.jsonl' in os.listdir(args.generated_path):
    logger.info('Using exisiting hard-negative data')
else: 
    logger.info('No hard-negative data found. Now mining it')
    miner = NegativeMiner(args.generated_path, prefix, retrievers=args.retrievers)
    miner.run()

if 'gpl-training-data.tsv' in os.listdir(args.generated_path):
    logger.info('Using existing GPL training data')
else:
    logger.info('No GPL training data found. Now generating it via pseudo labeling')
    pseudo_labeler = PseudoLabeler(args.generated_path, gen_queries, corpus, args.gpl_steps, args.batch_size_gpl, args.cross_encoder)
    pseudo_labeler.run()

ckpt_dir = os.path.join(args.output_dir, str(args.gpl_steps))
if not os.path.exists(ckpt_dir) or (os.path.exists(ckpt_dir) and not os.listdir(ckpt_dir)):
    logger.info('Now training GPL on generated data')
    #### Provide any HuggingFace model and fine-tune from scratch
    model_name = args.base_ckpt
    word_embedding_model = models.Transformer(model_name, max_seq_length=args.max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=args.pooling)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    fpath_gpl_data = os.path.join(args.generated_path, 'gpl-training-data.tsv')
    train_dataset = GenerativePseudoLabelingDataset(fpath_gpl_data, gen_queries, corpus)
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size_gpl, drop_last=True)  # Here shuffle=False, since (or assuming) we have done it in the pseudo labeling
    train_loss = MarginDistillationLoss(model=model)

    assert args.gpl_steps > 1000
    model.fit(
        [(train_dataloader, train_loss),],
        epochs=1,
        steps_per_epoch=args.gpl_steps,
        warmup_steps=1000,
        checkpoint_save_steps=10000,
        checkpoint_save_total_limit=10000,
        output_path=args.output_dir,
        checkpoint_path=args.output_dir,
        use_amp=args.use_amp
    )
else:
    logger.info('Trained GPL model found. Now skip training')


logger.info('Doing evaluation for GPL')
evaluate(args.evaluation_data, args.evaluation_output, ckpt_dir, args.max_seq_length, score_function='dot', pooling=args.pooling)

if args.mnrl_output_dir is not None:
    assert args.mnrl_evaluation_output is not None, "Evaluation path for MNRL should not be None, either"
    ckpt_dir = args.mnrl_output_dir
    if not os.path.exists(ckpt_dir) or (os.path.exists(ckpt_dir) and not os.listdir(ckpt_dir)):
        logger.info('Now training MNRL on generated data')
        mnrl(args.generated_path, args.base_ckpt, args.mnrl_output_dir, args.max_seq_length, args.use_amp)
    else:
        logger.info('Trained MNRL model found. Now skip training')

    logger.info('Doing evaluation for QGen (MNRL)')
    evaluate(args.evaluation_data, args.mnrl_evaluation_output, ckpt_dir, args.max_seq_length, score_function='cos_sim', pooling=args.pooling)
