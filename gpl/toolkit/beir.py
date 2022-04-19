import json
import os
from typing import Dict
import logging
logger = logging.getLogger(__name__)


def extract_queries_split(queries: Dict[str, str], qrels: Dict[str, Dict[str, float]]):
    qids_to_exclude = set(qrels) - set(queries)
    for qid in qids_to_exclude:
        queries.pop(qid)
    return queries


def save_queries(queries: Dict[str, str], output_dir):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'queries.jsonl')
    with open(save_path, 'w') as f:
        for qid, query in queries.items():
            line_dict = {"_id": qid, "text": query, "metadata": {}}
            line = json.dumps(line_dict) + '\n'
            f.write(line)
    
    logger.info(f'Saved (copied) queries into {save_path}')

def save_qrels(qrels: Dict[str, Dict[str, float]], output_dir, split):
    os.makedirs(os.path.join(output_dir, 'qrels'), exist_ok=True)

    assert split in ['train', 'test', 'dev']
    save_path = os.path.join(output_dir, f'qrels/{split}.tsv')    
    with open(save_path, 'w') as f:
        header = 'query-id\tcorpus-id\tscore\n'
        f.write(header)
        for qid, rels in qrels.items():
            for doc_id, score in rels.items():
                line = f'{qid}\t{doc_id}\t{score}\n'
                f.write(line)

    logger.info(f'Saved (copied) qrels into {save_path}')