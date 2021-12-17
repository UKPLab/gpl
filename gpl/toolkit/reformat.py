from logging import Logger
from random import choice
from sentence_transformers import SentenceTransformer, models
from transformers import BertModel, BertConfig, PreTrainedModel
from transformers.models.bert.modeling_bert import BertPooler
from gpl.toolkit import directly_loadable_by_sbert
import argparse
import logging
logger = logging.getLogger(__name__)


def simcse_like(model_name_or_path, output_path):
    ## For simcse-like models (e.g. "princeton-nlp/sup-simcse-bert-base-uncased"), the last layer is a linear layer on top of the CLS token embedding
    word_embedding_model = models.Transformer(model_name_or_path)
    auto_model = word_embedding_model.auto_model
    assert hasattr(auto_model, 'pooler'), 'This checkpoint (originally in HF-format) cannot be transformed into SBERT-format by this example method'
    pooler: BertPooler = auto_model.pooler  # This is the last layer, a linear mapping
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
    dense = models.Dense(
        in_features=pooler.dense.in_features, 
        out_features=pooler.dense.out_features, 
        init_weight=pooler.dense.weight, 
        init_bias=pooler.dense.bias,
        activation_function=pooler.activation
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense])
    assert directly_loadable_by_sbert(model), 'Cannot transform it into SBERT-format'
    model.save_model(output_path)

def dpr_like(model_name_or_path, output_path):
    ## For dpr-like models (e.g. "facebook/dpr-question_encoder-single-nq-base"), the last layer is also a linear layer on top of the CLS token embedding, but it does not output hidden states
    def down_to_pooler(model: PreTrainedModel):
        ## For "facebook/dpr-question_encoder-single-nq-base", the pooler is not directly accessible
        ## One needs to go through: DPRQuestionEncoder -> DPREncoder -> BertModel -> BertPooler
        if hasattr(model, 'pooler'):
            return model, model.pooler
        else:
            assert len(model._modules) == 1, 'This checkpoint does not have a pooler and thus cannot be transformed into SBERT-format in a naive way'
            next_module = list(model._modules)[0]
            return down_to_pooler(getattr(model, next_module))

    word_embedding_model = models.Transformer(model_name_or_path)
    auto_model = word_embedding_model.auto_model
    model_and_pooler = down_to_pooler(auto_model)  # Get the model and pooler
    auto_model: BertModel = model_and_pooler[0]
    ## TODO: Infer config class automatically    
    auto_model.config = BertConfig()  # Change DPRConfig into BertConfig
    pooler: BertPooler = model_and_pooler[1]
    word_embedding_model.auto_model = auto_model
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
    dense = models.Dense(
        in_features=pooler.dense.in_features, 
        out_features=pooler.dense.out_features, 
        init_weight=pooler.dense.weight, 
        init_bias=pooler.dense.bias,
        activation_function=pooler.activation
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense])
    assert directly_loadable_by_sbert(model), 'Cannot transform it into SBERT-format'
    model.save(output_path)

templates = {
    'simcse_like': simcse_like,
    'dpr_like': dpr_like
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', required=True, choices=list(templates.keys()), help='Selecting one template and do the reformatting (from HF-format to SBERT-format)')
    parser.add_argument('--model_name_or_path', help='The checkpoint in HF-format, which is going to be reformatted')
    parser.add_argument('--output_path', help='Output path to the reformatted checkpoint')
    args = parser.parse_args()
    templates[args.template](args.model_name_or_path, args.output_path)
    

