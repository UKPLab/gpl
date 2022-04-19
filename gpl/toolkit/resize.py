from math import log
from typing import List
from beir.datasets.data_loader import GenericDataLoader
import random
import os
import json
import logging
logger = logging.getLogger(__name__)


def resize(data_path, output_path, new_size, use_train_qrels:bool = False):
    train_qrels = None
    if use_train_qrels:
        corpus, _, train_qrels = GenericDataLoader(data_path).load(split='train')
    else:
        corpus = GenericDataLoader(data_path).load_corpus()
    
    if len(corpus) <= new_size:
        logger.warning('`new_size` should be smaller than the corpus size')
        corpus_new = list(corpus.items())
    else:
        corpus_new = random.sample(list(corpus.items()), k=new_size)
    corpus_new = dict(corpus_new)

    nadded = 0
    if train_qrels:
        for qid, rels in train_qrels.items():
            for doc_id in rels:
                if doc_id not in corpus_new:
                    assert doc_id in corpus
                    corpus_new[doc_id] = corpus[doc_id]
                    nadded += 1

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'corpus.jsonl'), 'w') as f:
        for doc_id, doc in corpus_new.items():
            doc['_id'] = doc_id
            f.write(json.dumps(doc) + '\n') 
    
    if not nadded:
        logger.info(f'Resized the corpus in {data_path} to {output_path} with new size {new_size}')
    else:
        logger.info(f'Resized the corpus in {data_path} to {output_path} with new size {new_size} + {nadded} (including the documents mentioned in the train qrels)')