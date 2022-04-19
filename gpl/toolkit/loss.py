from torch import nn, Tensor
from typing import Iterable, Dict
from torch.nn import functional as F
import logging
logger = logging.getLogger(__name__)


class MarginDistillationLoss(nn.Module):
    """
    Computes the MSE loss between the computed sentence embedding and a target sentence embedding. This loss
    is used when extending sentence embeddings to new languages as described in our publication
    Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation: https://arxiv.org/abs/2004.09813

    For an example, see the documentation on extending language models to new languages.
    """
    def __init__(self, model, scale: float = 1.0, similarity_fct = 'dot'):
        super(MarginDistillationLoss, self).__init__()
        self.model = model
        self.scale = scale
        assert similarity_fct in ['dot', 'cos_sim']
        self.similarity_fct = similarity_fct
        logger.info(f'Set GPL score function to {similarity_fct}')
        self.loss_fct = nn.MSELoss()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # sentence_features: query, positive passage, negative passage
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_query = reps[0]
        embeddings_pos = reps[1]
        embeddings_neg = reps[2]

        if self.similarity_fct == 'cosine':
            embeddings_query = F.normalize(embeddings_query, dim=-1)
            embeddings_pos = F.normalize(embeddings_pos, dim=-1)
            embeddings_neg = F.normalize(embeddings_neg, dim=-1)

        scores_pos = (embeddings_query * embeddings_pos).sum(dim=-1) * self.scale
        scores_neg = (embeddings_query * embeddings_neg).sum(dim=-1) * self.scale
        margin_pred = scores_pos - scores_neg

        return self.loss_fct(margin_pred, labels)