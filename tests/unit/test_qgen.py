from unittest import mock
from gpl.toolkit.qgen import qgen
from beir.generation import QueryGenerator
from beir.generation.models import QGenModel
from beir.datasets.data_loader import GenericDataLoader
import pytest


def test_qgen_oom():
    # Given:
    mock_generate = mock.patch.object(
        QueryGenerator,
        "generate",
        side_effect=RuntimeError("CUDA out of memory. Tried to allocate ..."),
    )
    mock_qgen_model_init = mock.patch.object(QGenModel, "__init__", return_value=None)
    mock_load_corpus = mock.patch.object(
        GenericDataLoader, "load_corpus", return_value=None
    )
    data_path = "dummy"
    output_dir = "dummy"
    hints = [
        "CUDA out of memory during query generation",
        "batch_size_generation:",
        "queries_per_passage:",
        "Please try smaller `queries_per_passage` and/or `batch_size_generation`.",
    ]

    # When and then:
    with mock_generate, mock_qgen_model_init, mock_load_corpus:
        with pytest.raises(RuntimeError):
            qgen(data_path, output_dir)

        try:
            qgen(data_path, output_dir)
        except RuntimeError as e:
            for hint in hints:
                assert hint in str(e)
