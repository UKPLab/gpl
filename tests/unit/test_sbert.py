from sentence_transformers import SentenceTransformer
from unittest import mock
from gpl.toolkit.sbert import directly_loadable_by_sbert
import pytest


def test_loadable_by_sbert_oom(sbert: SentenceTransformer):
    with mock.patch.object(SentenceTransformer, "encode") as sbert_encode:
        sbert_encode.side_effect = RuntimeError(
            "CUDA out of memory. Tried to allocate ..."
        )
        with pytest.raises(RuntimeError):
            directly_loadable_by_sbert(sbert)
