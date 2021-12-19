from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
from sentence_transformers import SentenceTransformer, losses, models
from .sbert import load_sbert
import os


# TODO: `sep` argument
def mnrl(data_path, base_ckpt, output_dir, max_seq_length=350, use_amp=True, qgen_prefix='qgen', pooling=None):
    #### Training on Generated Queries ####
    corpus, gen_queries, gen_qrels = GenericDataLoader(data_path, prefix=qgen_prefix).load(split="train")

    #### Provide any HuggingFace model and fine-tune from scratch
    model = load_sbert(base_ckpt, pooling, max_seq_length)

    #### Provide any sentence-transformers model path
    retriever = TrainRetriever(model=model, batch_size=75)

    #### Prepare training samples
    #### The current version of QGen training in BeIR use fixed separation token ' ' !!!
    train_samples = retriever.load_train(corpus, gen_queries, gen_qrels)
    train_dataloader = retriever.prepare_train(train_samples, shuffle=True)
    train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)

    #### Provide model save path
    model_save_path = output_dir
    os.makedirs(model_save_path, exist_ok=True)

    #### Configure Train params
    num_epochs = 1
    warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

    retriever.fit(
        train_objectives=[(train_dataloader, train_loss)], 
        epochs=num_epochs,
        output_path=model_save_path,
        warmup_steps=warmup_steps,
        use_amp=use_amp
    )
