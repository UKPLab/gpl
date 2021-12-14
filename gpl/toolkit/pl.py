import json
from sentence_transformers import CrossEncoder
from .dataset import HardNegativeDataset
from torch.utils.data import DataLoader
import tqdm
import os
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

def hard_negative_collate_fn(batch):
    query_id, pos_id, neg_id = zip(*[example.guid for example in batch])
    query, pos, neg = zip(*[example.texts for example in batch])
    return (query_id, pos_id, neg_id), (query, pos, neg)


class PseudoLabeler(object):

    def __init__(self, generated_path, gen_queries, corpus, total_steps, batch_size, cross_encoder):
        assert 'hard-negatives.jsonl' in os.listdir(generated_path)
        fpath_hard_negatives = os.path.join(generated_path, 'hard-negatives.jsonl')
        self.cross_encoder = CrossEncoder(cross_encoder)
        hard_negative_dataset = HardNegativeDataset(fpath_hard_negatives, gen_queries, corpus)    
        self.hard_negative_dataloader = DataLoader(hard_negative_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
        self.hard_negative_dataloader.collate_fn = hard_negative_collate_fn
        self.output_path = os.path.join(generated_path, 'gpl-training-data.tsv')
        self.total_steps = total_steps
    
    def run(self):
        # header: 'query_id', 'positive_id', 'negative_id', 'pseudo_label_margin'
        data = []

        hard_negative_iterator = iter(self.hard_negative_dataloader)
        logger.info('Begin pseudo labeling')
        for _ in tqdm.trange(self.total_steps):
            try:
                batch = next(hard_negative_iterator)
            except StopIteration:
                hard_negative_iterator = iter(self.hard_negative_dataloader)
                batch = next(hard_negative_iterator)

            (query_id, pos_id, neg_id), (query, pos, neg) = batch
            scores = self.cross_encoder.predict(
                list(zip(query, pos)) + list(zip(query, neg)), 
                show_progress_bar=False
            )
            labels = scores[:len(query)] - scores[len(query):]
            
            batch_gpl = map(lambda quad: '\t'.join((*quad[:3], str(quad[3]))) + '\n', zip(query_id, pos_id, neg_id, labels))
            data.extend(batch_gpl)
        
        logger.info('Done pseudo labeling and saving data')
        with open(self.output_path, 'w') as f:
            f.writelines(data)
        
        logger.info(f'Saved GPL training data to {self.output_path}')
    