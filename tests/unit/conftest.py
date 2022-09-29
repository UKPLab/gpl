import os
import pytest
import tempfile
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.configuration_bert import BertConfig
from sentence_transformers import SentenceTransformer
import shutil


@pytest.fixture(name="sbert_path", scope="session")
def sbert_path_fixture() -> str:
    try:
        local_dir = tempfile.mkdtemp()

        vocab = [
            "[PAD]",
            "[UNK]",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "the",
            "of",
            "and",
            "in",
            "to",
            "was",
            "he",
        ]
        vocab_file = os.path.join(local_dir, "vocab.txt")
        with open(vocab_file, "w") as f:
            f.write("\n".join(vocab))

        config = BertConfig(
            vocab_size=len(vocab),
            hidden_size=2,
            num_attention_heads=1,
            num_hidden_layers=2,
            intermediate_size=2,
            max_position_embeddings=10,
        )

        bert = BertModel(config)
        tokenizer = BertTokenizer(vocab_file)

        bert.save_pretrained(local_dir)
        tokenizer.save_pretrained(local_dir)

        yield local_dir
    finally:
        shutil.rmtree(local_dir)
        print("Cleared temporary SBERT/BERT model")


@pytest.fixture(name="sbert", scope="session")
def sbert_fixture(sbert_path: str) -> SentenceTransformer:
    yield SentenceTransformer(sbert_path)
