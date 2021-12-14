from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
import sentence_transformers
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import os
import logging
import numpy as np 
import json
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

def evaluate(data_path, output_dir, model_name_or_path, max_seq_length, score_function, pooling):
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

        if '0_Transformer' in os.listdir(model_name_or_path):
            model_name_or_path = os.path.join(model_name_or_path, '0_Transformer')
        word_embedding_model = sentence_transformers.models.Transformer(model_name_or_path, max_seq_length=max_seq_length)
        pooling_model = sentence_transformers.models.Pooling(
            word_embedding_model.get_word_embedding_dimension(), 
            pooling
        )
        model = sentence_transformers.SentenceTransformer(modules=[word_embedding_model, pooling_model])
        sbert = models.SentenceBERT(sep=' ')
        sbert.q_model = model
        sbert.doc_model = model
        
        model = DRES(sbert, batch_size=16)
        assert score_function in ['dot', 'cos_sim']
        retriever = EvaluateRetrieval(model, score_function=score_function) # or "dot" for dot-product
        results = retriever.retrieve(corpus, queries)

        #### Evaluate your retrieval using NDCG@k, MAP@K ...
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, [10,])
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

if __name__ == '__name__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path')
    parser.add_argument('--output_dir')
    parser.add_argument('--model_name_or_path')
    parser.add_argument('--max_seq_length', type=int)
    parser.add_argument('--score_function', choices=['dot', 'cos_sim'])
    args = parser.parse_args()

    evaluate(args.data_path, args.output_dir, args.model_name_or_path, args.max_seq_length, args.score_function)
