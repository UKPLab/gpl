from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
from sentence_transformers import SentenceTransformer, losses, models
import os


def mnrl(data_path, base_ckpt, output_dir, max_seq_length=350, use_amp=True, qgen_prefix='qgen'):
    prefix = qgen_prefix
    #### Training on Generated Queries ####
    corpus, gen_queries, gen_qrels = GenericDataLoader(data_path, prefix=prefix).load(split="train")

    #### Provide any HuggingFace model and fine-tune from scratch
    model_name = base_ckpt
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    #### Provide any sentence-transformers model path
    retriever = TrainRetriever(model=model, batch_size=75)

    #### Prepare training samples
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
