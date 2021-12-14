from beir.datasets.data_loader import GenericDataLoader
import random
import os
import json
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

def resize(data_path, output_path, new_size):
    corpus = GenericDataLoader(data_path).load_corpus()
    corpus_new = random.sample(list(corpus.items()), k=new_size)
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'corpus.jsonl'), 'w') as f:
        for line in corpus_new:
            _id, doc = line
            doc['_id'] = _id
            f.write(json.dumps(doc) + '\n') 
    logger.info(f'Resized the corpus in {data_path} to {output_path} with new size {new_size}')