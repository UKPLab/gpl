from ast import parse
import json
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.generation import QueryGenerator as QGen
from beir.generation.models import QGenModel
from beir.retrieval.train import TrainRetriever
from sentence_transformers import SentenceTransformer, losses, models, util
import torch
from sentence_transformers import CrossEncoder
from torch.nn import functional as F
from easy_elasticsearch import ElasticSearchBM25
import tqdm
import numpy as np
import os
import logging
import argparse
import time


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)


class NegativeMiner(object):

    def __init__(self, generated_path, prefix, retrievers=['bm25', 'msmarco-distilbert-base-v3', 'msmarco-MiniLM-L-6-v3']):
        self.corpus, self.gen_queries, self.gen_qrels = GenericDataLoader(generated_path, prefix=prefix).load(split="train")
        self.output_path = os.path.join(generated_path, 'hard-negatives.jsonl')
        self.retrievers = retrievers

    def _get_doc(self, did):
        return ' '.join([self.corpus[did]['title'], self.corpus[did]['text']])
    
    def _mine_sbert(self, model_name, nneg=50):
        logger.info(f'Mining with {model_name}')
        result = {}
        sbert = SentenceTransformer(model_name)
        docs = list(map(self._get_doc, self.corpus.keys()))
        dids = np.array(list(self.corpus.keys()))
        doc_embs = sbert.encode(
            docs, 
            batch_size=128, 
            show_progress_bar=True, 
            convert_to_numpy=False, 
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        qids = list(self.gen_qrels.keys())
        queries = list(map(lambda qid: self.gen_queries[qid], qids))
        for start in tqdm.trange(0, len(queries), 128):
            qid_batch = qids[start:start+128]
            qemb_batch = sbert.encode(
                queries[start:start+128], 
                show_progress_bar=False, 
                convert_to_numpy=False, 
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            score_mtrx = torch.matmul(qemb_batch, doc_embs.t())  # (qsize, dsize)
            _, indices_topk = score_mtrx.topk(k=nneg, dim=-1)
            indices_topk = indices_topk.tolist()
            for qid, neg_dids in zip(qid_batch, indices_topk):
                neg_dids = dids[neg_dids].tolist()
                for pos_did in self.gen_qrels[qid]:
                    if pos_did in neg_dids:
                        neg_dids.remove(pos_did)
                result[qid] = neg_dids
        return result
    
    def _mine_bm25(self, nneg=50):
        logger.info(f'Mining with bm25')
        result = {}
        docs = list(map(self._get_doc, self.corpus.keys()))
        dids = np.array(list(self.corpus.keys()))
        pool = dict(zip(dids, docs))
        bm25 = ElasticSearchBM25(pool, port_http='9222', port_tcp='9333', service_type='executable', index_name=f'one_trial{int(time.time() * 1000000)}')
        for qid, pos_dids in tqdm.tqdm(self.gen_qrels.items()):
            query = self.gen_queries[qid]
            rank = bm25.query(query, topk=nneg)  # topk should be <= 10000
            neg_dids = list(rank.keys())
            for pos_did in self.gen_qrels[qid]:
                if pos_did in neg_dids:
                    neg_dids.remove(pos_did)
            result[qid] = neg_dids
        return result

    def run(self):
        hard_negatives = {}
        for retriever in self.retrievers:
            if retriever == 'bm25':
                hard_negatives[retriever] = self._mine_bm25()
            else:
                hard_negatives[retriever] = self._mine_sbert(retriever)
        
        logger.info('Combining all the data')
        result_jsonl = []
        for qid, pos_dids in tqdm.tqdm(self.gen_qrels.items()):
            line = {
                'qid': qid,
                'pos': list(pos_dids.keys()),
                'neg': {
                    k: v[qid] for k, v in hard_negatives.items()
                }
            }
            result_jsonl.append(line)

        logger.info(f'Saving data to {self.output_path}')
        with open(self.output_path, 'w') as f:
            for line in result_jsonl:
                f.write(json.dumps(line) + '\n')
        logger.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated_path')
    args = parser.parse_args()

    miner = NegativeMiner(args.generated_path, 'gen')
    miner.run()