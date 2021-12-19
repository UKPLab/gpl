from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
import sentence_transformers
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import os
import logging
import numpy as np 
import json
from typing import List
import argparse
from .sbert import load_sbert
logger = logging.getLogger(__name__)


def evaluate(
    data_path: str, 
    output_dir: str, 
    model_name_or_path: str, 
    max_seq_length: int = 350, 
    score_function: str = 'dot', 
    pooling: str = None, 
    sep: str = ' ', 
    k_values: List[int] = [10,]
):
    data_paths = []
    if 'cqadupstack' in data_path:
        data_paths = [os.path.join(data_path, sub_dataset) for sub_dataset in \
            ['android', 'english', 'gaming', 'gis', 'mathematica', 'physics', 'programmers', 'stats', 'tex', 'unix', 'webmasters', 'wordpress',]]
    else:
        data_paths.append(data_path)

    ndcgs = []
    _maps = []
    recalls = []
    precisions = []
    mrrs = []
    for data_path in data_paths:
        corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

        model = load_sbert(model_name_or_path, pooling, max_seq_length)
        sbert = models.SentenceBERT(sep=sep)
        sbert.q_model = model
        sbert.doc_model = model
        
        model = DRES(sbert, batch_size=16)
        assert score_function in ['dot', 'cos_sim']
        retriever = EvaluateRetrieval(model, score_function=score_function) # or "dot" for dot-product
        results = retriever.retrieve(corpus, queries)

        #### Evaluate your retrieval using NDCG@k, MAP@K ...
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, k_values)
        mrr = EvaluateRetrieval.evaluate_custom(qrels, results, [10,], metric='mrr')
        ndcgs.append(ndcg)
        _maps.append(_map)
        recalls.append(recall)
        precisions.append(precision)
        mrrs.append(mrr)

    ndcg = {k: np.mean([score[k] for score in ndcgs]) for k in ndcg}
    _map = {k: np.mean([score[k] for score in _maps]) for k in _map}
    recall = {k: np.mean([score[k] for score in recalls]) for k in recall}
    precision = {k: np.mean([score[k] for score in precisions]) for k in precision}
    mrr = {k: np.mean([score[k] for score in mrrs]) for k in mrr}

    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, 'results.json')
    with open(result_path, 'w') as f:
        json.dump({
            'ndcg': ndcg,
            'map': _map,
            'recall': recall,
            'precicion': precision,
            'mrr': mrr
        }, f, indent=4)
    logger.info(f'Saved evaluation results to {result_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--output_dir')
    parser.add_argument('--model_name_or_path')
    parser.add_argument('--max_seq_length', type=int)
    parser.add_argument('--score_function', choices=['dot', 'cos_sim'])
    parser.add_argument('--pooling', choices=['dot', 'cos_sim'])
    parser.add_argument('--sep', type=str, default=' ', help="Separation token between title and body text for each passage. The concatenation way is `sep.join([title, body])`")
    parser.add_argument('--k_values', nargs='+', type=int, default=[10,], help="The K values in the evaluation. This will compute nDCG@K, recall@K, precision@K and MAP@K")    
    args = parser.parse_args()
    evaluate(**vars(args))
